using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

namespace SPHFluid
{
    public class SPHSolver
    {
        #region Smooth Kernels
        public static readonly double kPoly6Const = 1.566681471061;
        public static readonly double gradKPoly6Const = -9.4000888264;
        public static readonly double lapKPoly6Const = -9.4000888264;
        public static readonly double kSpikyConst = 4.774648292757;
        public static readonly double gradKSpikyConst = -14.3239448783;
        public static readonly double kViscosityConst = 2.387324146378;
        public static readonly double lapkViscosityConst = 14.3239448783;

        public double KernelPoly6(Vector3d r)
        {
            double sqrDiff = (kr2 - r.sqrMagnitude);
            if (sqrDiff < 0)
                return 0;
            return kPoly6Const * inv_kr9 * sqrDiff * sqrDiff * sqrDiff;
        }

        public Vector3d GradKernelPoly6(Vector3d r)
        {
            double sqrDiff = (kr2 - r.sqrMagnitude);
            if (sqrDiff < 0)
                return Vector3d.zero;
            return gradKPoly6Const * inv_kr9 * sqrDiff * sqrDiff * r;
        }

        public double LaplacianKernelPoly6(Vector3d r)
        {
            double r2 = r.sqrMagnitude;
            double sqrDiff = (kr2 - r2);
            if (sqrDiff < 0)
                return 0;
            return lapKPoly6Const * inv_kr9 * sqrDiff * (3 * kr2 - 7 * r2);
        }

        public double KernelSpiky(Vector3d r)
        {
            double diff = kernelRadius - r.magnitude;
            if (diff < 0)
                return 0;
            return kSpikyConst * inv_kr6 * diff * diff * diff;
        }

        public Vector3d GradKernelSpiky(Vector3d r)
        {
            double mag = r.magnitude;
            double diff = (kernelRadius - mag);
            if (diff < 0 || mag <= 0)
                return Vector3d.zero;
            r *= (1 / mag);
            return gradKSpikyConst * inv_kr6 * diff * diff * r;
        }

        public double KernelViscosity(Vector3d r)
        {
            double mag = r.magnitude;
            if (kernelRadius - mag < 0)
                return 0;
            double sqrMag = mag * mag;
            return kViscosityConst * inv_kr3 * (-0.5 * mag * sqrMag * inv_kr3 + sqrMag / (kr2) + 0.5 * kernelRadius / mag - 1);
        }

        public double LaplacianKernelViscosity(Vector3d r)
        {
            double diff = kernelRadius - r.magnitude;
            if (diff < 0)
                return 0;
            return lapkViscosityConst * inv_kr6 * diff;
        }
        #endregion

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
        public SPHGridCell[] grid;

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
        private int _sphthreadGroupNum;

        public ComputeBuffer _bufferParticles;
        private ComputeBuffer _bufferNeighborSpace;
        public ComputeBuffer _bufferParticleNumPerCell;
        private ComputeBuffer _bufferParticlePrefixLocalOffset;
        private bool[] _neighborSpaceInit;
        private int[] _particleNumPerCellInit;

        public const int scanCellNumThreadGroupSize = 512;
        private int _scanThreadGroupNum;

        private ComputeShader _shaderRadixSort;
        private GPURadixSortParticles _GPUSorter;

        #endregion

        public SPHSolver(int maxParticleNum, double timeStep, double kernelRadius,
                        double stiffness, double restDensity, Vector3d externalAcc,
                        double viscosity, double tensionCoef, double surfaceThreshold,
                        int gridSizeX, int gridSizeY, int gridSizeZ, ComputeShader shaderSPH,
                        ComputeShader shaderRadixSort)
        {
            this.maxParticleNum = maxParticleNum;

            allCSParticles = new List<CSParticle>();
            allCSParticles.Capacity = maxParticleNum;

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
            grid = new SPHGridCell[gridCountXYZ];
            for (int x = 0; x < gridSize._x; ++x)
                for (int y = 0; y < gridSize._y; ++y)
                    for (int z = 0; z < gridSize._z; ++z)
                    {
                        grid[x * gridCountYZ + y * gridSize._z + z] = new SPHGridCell(x, y, z);
                        //grid[x * gridCountYZ + y * gridSize._z + z].particles.Capacity = maxParticleNum; //maybe not a good idea
                    }
            
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
            _neighborSpaceInit = new bool[currParticleNum * 27];
            _bufferNeighborSpace.SetData(_neighborSpaceInit);

            _bufferParticleNumPerCell = new ComputeBuffer(gridCountXYZ, sizeof(int));
            _particleNumPerCellInit = new int[gridCountXYZ];
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

            //_bufferParticleNumPerCell.GetData(_particleStartIndexPerCell);
            //int startIdx = 0;
            //for (int i = 0; i < gridCountXYZ + 1; ++i)
            //{
            //    int oldVal = _particleStartIndexPerCell[i];
            //    _particleStartIndexPerCell[i] = startIdx;
            //    startIdx += oldVal;
            //}
            //_bufferParticleStartIndexPerCell.SetData(_particleStartIndexPerCell);

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

            _bufferParticles.GetData(_allCSParticlesContainer);
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

            _shaderSPH.Dispatch(_kernelAp, _sphthreadGroupNum, 1, 1);

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

            _bufferParticles.GetData(_allCSParticlesContainer);
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

            _GPUSorter.Free();
        }
    }
}
