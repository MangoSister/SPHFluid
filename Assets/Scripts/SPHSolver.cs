using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

namespace SPHFluid
{
    public class SPHSolver
    {
        public int maxParticleNum;
        public int currParticleNum { get { return allCSParticles.Count; } }

        public double timeStep;
        public double kernelRadius;
        #region Precomputed Values
        public double kr2, inv_kr3, inv_kr6, inv_kr9;
        #endregion
        public double stiffness;
        public double restDensity;
        public Vector3d externalAcc;
        public double viscosity;
        public double tensionCoef;
        public double surfaceThreshold;

        public Int3 gridSize;
        public int gridCountXYZ, gridCountXY, gridCountXZ, gridCountYZ;

        #region GPU Adaptation
        public List<CSParticle> allCSParticles;
        public CSParticle[] _allCSParticlesContainer;
        private ComputeShader _shaderSPH;
        private int _kernelCci;
        private int _kernelScanLocal;
        private int _kernelScanGlobal;
        private int _kernelFns;
        private int _kernelUpd;
        private int _kernelUpff;
        private int _kernelAp;
        private int _kernelIp;

        public const int sphThreadGroupSize = 512;
        public int _sphthreadGroupNum;

        public ComputeBuffer _bufferParticles;
        private ComputeBuffer _bufferNeighborSpace;
        public ComputeBuffer _bufferParticleNumPerCell;
        private ComputeBuffer _bufferParticlePrefixLocalOffset;

        private int[] _neighborSpaceInit;
        private int[] _particleNumPerCellInit;

        public const int scanCellNumThreadGroupSize = 512;
        private int _scanThreadGroupNum;

        private ComputeShader _shaderRadixSort;
        private GPURadixSortParticles _GPUSorter;

        public List<CSSphere> _obstacles;
        private ComputeBuffer _bufferObstacles;

        #endregion 

        public SPHSolver(int maxParticleNum, double timeStep, double kernelRadius,
                        double stiffness, double restDensity, Vector3d externalAcc,
                        double viscosity, double tensionCoef, double surfaceThreshold,
                        int gridSizeX, int gridSizeY, int gridSizeZ, ComputeShader shaderSPH,
                        ComputeShader shaderRadixSort)
        {
            this.maxParticleNum = maxParticleNum;

            allCSParticles = new List<CSParticle>();
            //allCSParticles.Capacity = maxParticleNum;

            this.timeStep = timeStep;
            this.kernelRadius = kernelRadius;

            kr2 = kernelRadius * kernelRadius;
            inv_kr3 = 1 / (kernelRadius * kr2);
            inv_kr6 = inv_kr3 * inv_kr3;
            inv_kr9 = inv_kr6 * inv_kr3;

            this.stiffness = stiffness;
            this.restDensity = restDensity;
            this.externalAcc = externalAcc;
            this.viscosity = viscosity;
            this.tensionCoef = tensionCoef;
            this.surfaceThreshold = surfaceThreshold;

            gridSize = new Int3(gridSizeX, gridSizeY, gridSizeZ);
            gridCountXYZ = gridSizeX * gridSizeY * gridSizeZ;
            gridCountXY = gridSizeX * gridSizeY;
            gridCountXZ = gridSizeX * gridSizeZ;
            gridCountYZ = gridSizeY * gridSizeZ;
            
            _shaderSPH = shaderSPH;

            _kernelCci = _shaderSPH.FindKernel("ComputeCellIdx");
            _kernelScanLocal = _shaderSPH.FindKernel("ScanCellNumLocal");
            _kernelScanGlobal = _shaderSPH.FindKernel("ScanCellNumGlobal");
            _kernelFns = _shaderSPH.FindKernel("FindNeighborSpace");
            _kernelUpd = _shaderSPH.FindKernel("UpdatePressureDensity");
            _kernelUpff = _shaderSPH.FindKernel("UpdateParticleFluidForce");
            _kernelAp = _shaderSPH.FindKernel("AdvanceParticle");
            _kernelIp = _shaderSPH.FindKernel("InitParticle");

            _shaderSPH.SetFloat("_TimeStep", (float)timeStep);
            _shaderSPH.SetInts("_SphGridSize", gridSize._x, gridSize._y, gridSize._z);
            _shaderSPH.SetFloat("_KernelRadius", (float)kernelRadius);
            _shaderSPH.SetFloat("_inv_KernelRadius", 1f / (float)kernelRadius);
            _shaderSPH.SetFloat("_kr2", (float)kr2);
            _shaderSPH.SetFloat("_inv_kr3", (float)inv_kr3);
            _shaderSPH.SetFloat("_inv_kr6", (float)inv_kr6);
            _shaderSPH.SetFloat("_inv_kr9", (float)inv_kr9);
            _shaderSPH.SetFloat("_Stiffness", (float)stiffness);
            _shaderSPH.SetFloat("_RestDensity", (float)restDensity);
            _shaderSPH.SetFloats("_ExternalAcc", (float)externalAcc.x, (float)externalAcc.y, (float)externalAcc.z);
            _shaderSPH.SetFloat("_Viscosity", (float)viscosity);
            _shaderSPH.SetFloat("_TensionCoef", (float)tensionCoef);
            _shaderSPH.SetFloat("_SurfaceThreshold", (float)surfaceThreshold);
            _shaderSPH.SetFloat("_Eps", (float)MathHelper.Eps);

            _shaderRadixSort = shaderRadixSort;
        }

        public bool CreateGPUParticle(float mass, Vector3 initPos, Vector3 initVelo)
        {
            if (currParticleNum >= maxParticleNum)
                return false;

            allCSParticles.Add(new CSParticle(mass, initPos, initVelo));

            return true;
        }

        public void InitOnGPU()
        {
            _sphthreadGroupNum = Mathf.CeilToInt((float)currParticleNum / (float)sphThreadGroupSize);
            _scanThreadGroupNum = Mathf.CeilToInt((float)gridCountXYZ / (float)scanCellNumThreadGroupSize);
            _GPUSorter = new GPURadixSortParticles(_shaderRadixSort, currParticleNum);

            _shaderSPH.SetInt("_ParticleNum", currParticleNum);

            _allCSParticlesContainer = allCSParticles.ToArray();
            _bufferParticles = new ComputeBuffer(currParticleNum, CSParticle.stride);
            _bufferParticles.SetData(_allCSParticlesContainer);

            _bufferNeighborSpace = new ComputeBuffer(currParticleNum * 27, 4);
            _neighborSpaceInit = new int[currParticleNum * 27];
            _bufferNeighborSpace.SetData(_neighborSpaceInit);

            _bufferParticleNumPerCell = new ComputeBuffer(gridCountXYZ + 1, sizeof(int));
            _particleNumPerCellInit = new int[gridCountXYZ + 1];
            _bufferParticleNumPerCell.SetData(_particleNumPerCellInit);


            _bufferParticlePrefixLocalOffset = new ComputeBuffer(_scanThreadGroupNum , sizeof(int));

            _shaderSPH.SetBuffer(_kernelCci, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelCci, "_Particles", _bufferParticles);

            _shaderSPH.Dispatch(_kernelCci, _sphthreadGroupNum, 1, 1);

            _GPUSorter.Init(_bufferParticles);
            _GPUSorter.Sort();

            _shaderSPH.SetBuffer(_kernelScanLocal, "_ParticlePrefixLocalOffset", _bufferParticlePrefixLocalOffset);
            _shaderSPH.SetBuffer(_kernelScanLocal, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.Dispatch(_kernelScanLocal, _scanThreadGroupNum, 1, 1);

            _shaderSPH.SetBuffer(_kernelScanGlobal, "_ParticlePrefixLocalOffset", _bufferParticlePrefixLocalOffset);
            _shaderSPH.SetBuffer(_kernelScanGlobal, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.Dispatch(_kernelScanGlobal, _scanThreadGroupNum, 1, 1);

            _shaderSPH.SetBuffer(_kernelFns, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelFns, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelFns, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelFns, _sphthreadGroupNum, 1, 1);

            _shaderSPH.SetBuffer(_kernelUpd, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelUpd, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelUpd, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelUpd, _sphthreadGroupNum, 1, 1);

            _shaderSPH.SetBuffer(_kernelUpff, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelUpff, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelUpff, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelUpff, _sphthreadGroupNum, 1, 1);

            _shaderSPH.SetBuffer(_kernelIp, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelIp, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelIp, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelIp, _sphthreadGroupNum, 1, 1);

            //_bufferParticles.GetData(_allCSParticlesContainer);
        }

        public void StepOnGPU()
        {
            _shaderSPH.SetBuffer(_kernelFns, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelFns, "_Particles", _bufferParticles);

            _bufferNeighborSpace.SetData(_neighborSpaceInit);
            _shaderSPH.SetBuffer(_kernelFns, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelFns, _sphthreadGroupNum, 1, 1);

            _shaderSPH.SetBuffer(_kernelUpd, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelUpd, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelUpd, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelUpd, _sphthreadGroupNum, 1, 1);

            _shaderSPH.SetBuffer(_kernelUpff, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelUpff, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelUpff, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelUpff, _sphthreadGroupNum, 1, 1);

            _shaderSPH.SetBuffer(_kernelAp, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelAp, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelAp, "_NeighborSpace", _bufferNeighborSpace);

            if (_obstacles != null && _obstacles.Count > 0)
            {
                _bufferObstacles = new ComputeBuffer(_obstacles.Count, CSSphere.stride);
                _bufferObstacles.SetData(_obstacles.ToArray());
                _shaderSPH.SetBuffer(_kernelAp, "_Obstacles", _bufferObstacles);
            }
            
            _shaderSPH.Dispatch(_kernelAp, _sphthreadGroupNum, 1, 1);

            if (_bufferObstacles != null)
            {
                _bufferObstacles.Release();
                _bufferObstacles = null;
            }

            _bufferParticleNumPerCell.SetData(_particleNumPerCellInit);
            _shaderSPH.SetBuffer(_kernelCci, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelCci, "_Particles", _bufferParticles);

            _shaderSPH.Dispatch(_kernelCci, _sphthreadGroupNum, 1, 1);

            _GPUSorter.Init(_bufferParticles);
            _GPUSorter.Sort();

            _shaderSPH.SetBuffer(_kernelScanLocal, "_ParticlePrefixLocalOffset", _bufferParticlePrefixLocalOffset);
            _shaderSPH.SetBuffer(_kernelScanLocal, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.Dispatch(_kernelScanLocal, _scanThreadGroupNum, 1, 1);

            _shaderSPH.SetBuffer(_kernelScanGlobal, "_ParticlePrefixLocalOffset", _bufferParticlePrefixLocalOffset);
            _shaderSPH.SetBuffer(_kernelScanGlobal, "_ParticleCellNumPrefixSum", _bufferParticleNumPerCell);
            _shaderSPH.Dispatch(_kernelScanGlobal, _scanThreadGroupNum, 1, 1);


            //_bufferParticles.GetData(_allCSParticlesContainer);
        }

        public void Free()
        {
            if (_bufferNeighborSpace != null)
            {
                _bufferNeighborSpace.Release();
                _bufferNeighborSpace = null;
            }

            if (_bufferParticleNumPerCell != null)
            {
                _bufferParticleNumPerCell.Release();
                _bufferParticleNumPerCell = null;
            }

            if (_bufferParticles != null)
            {
                _bufferParticles.Release();
                _bufferParticles = null;
            }

            if (_bufferParticlePrefixLocalOffset != null)
            {
                _bufferParticlePrefixLocalOffset.Release();
                _bufferParticlePrefixLocalOffset = null;
            }

            if (_bufferObstacles != null)
            {
                _bufferObstacles.Release();
                _bufferObstacles = null;
            }

            _GPUSorter.Free();
        }
    }
}
