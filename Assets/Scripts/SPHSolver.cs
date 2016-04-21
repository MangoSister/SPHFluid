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
        public List<SPHParticle> allParticles { get; private set; }

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

        public bool isSolving { get; private set; }

        #region GPU Adaptation
        public List<CSParticle> allCSParticles;
        public CSParticle[] _allCSParticlesContainer;
        private ComputeShader _shaderSPH;
        private int _kernelCci;
        private int _kernelFns;
        private int _kernelUpd;
        private int _kernelUpff;
        private int _kernelAp;
        private int _kernelIp;

        public ComputeBuffer _bufferParticles;
        private ComputeBuffer _bufferNeighborSpace;
        private ComputeBuffer _bufferParticleNumPerCell;
        public ComputeBuffer _bufferParticleStartIndexPerCell;
        public int[] _particleStartIndexPerCell;
        private bool[] _neighborSpaceInit;
        private int[] _particleNumPerCellInit;

        #endregion

        public SPHSolver(int maxParticleNum, double timeStep, double kernelRadius,
                        double stiffness, double restDensity, Vector3d externalAcc,
                        double viscosity, double tensionCoef, double surfaceThreshold,
                        int gridSizeX, int gridSizeY, int gridSizeZ, ComputeShader shaderSPH)
        {
            this.maxParticleNum = maxParticleNum;
            allParticles = new List<SPHParticle>();
            allParticles.Capacity = maxParticleNum;
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

            isSolving = false;
        }

        public Int3 FindCellIdx(Vector3d pos)
        {
            pos /= kernelRadius;
            return new Int3((int)Math.Floor(pos.x), (int)Math.Floor(pos.y), (int)Math.Floor(pos.z));
        }

        public List<Int3> FindNeighborSpace(Int3 idx)
        {
            if (idx._x < 0 || idx._x >= gridSize._x ||
                idx._y < 0 || idx._y >= gridSize._y ||
                idx._z < 0 || idx._z >= gridSize._z)
                return null;

            List<Int3> output = new List<Int3>() { idx };
            if (idx._x > 0)
                output.Add(new Int3(idx._x - 1, idx._y, idx._z));
            if (idx._y > 0)
                output.Add(new Int3(idx._x, idx._y - 1, idx._z));
            if (idx._z > 0)
                output.Add(new Int3(idx._x, idx._y, idx._z - 1));

            if (idx._x < gridSize._x - 1)
                output.Add(new Int3(idx._x + 1, idx._y, idx._z));
            if (idx._y < gridSize._y - 1)
                output.Add(new Int3(idx._x, idx._y + 1, idx._z));
            if (idx._z < gridSize._z - 1)
                output.Add(new Int3(idx._x, idx._y, idx._z + 1));

            if (idx._x > 0 && idx._y > 0)
                output.Add(new Int3(idx._x - 1, idx._y - 1, idx._z));
            if (idx._y > 0 && idx._z > 0)
                output.Add(new Int3(idx._x, idx._y - 1, idx._z - 1));
            if (idx._x > 0 && idx._z > 0)
                output.Add(new Int3(idx._x - 1, idx._y, idx._z - 1));

            if (idx._x < gridSize._x - 1 && idx._y < gridSize._y - 1)
                output.Add(new Int3(idx._x + 1, idx._y + 1, idx._z));
            if (idx._y < gridSize._y - 1 && idx._z < gridSize._z - 1)
                output.Add(new Int3(idx._x, idx._y + 1, idx._z + 1));
            if (idx._x < gridSize._x - 1 && idx._z < gridSize._z - 1)
                output.Add(new Int3(idx._x + 1, idx._y, idx._z + 1));

            if (idx._x > 0 && idx._y < gridSize._y - 1)
                output.Add(new Int3(idx._x - 1, idx._y + 1, idx._z));
            if (idx._y > 0 && idx._z < gridSize._z - 1)
                output.Add(new Int3(idx._x, idx._y - 1, idx._z + 1));
            if (idx._x > 0 && idx._z < gridSize._z - 1)
                output.Add(new Int3(idx._x - 1, idx._y, idx._z + 1));

            if (idx._x < gridSize._x - 1 && idx._y > 0)
                output.Add(new Int3(idx._x + 1, idx._y - 1, idx._z));
            if (idx._y < gridSize._y - 1 && idx._z > 0)
                output.Add(new Int3(idx._x, idx._y + 1, idx._z - 1));
            if (idx._x < gridSize._x - 1 && idx._z > 0)
                output.Add(new Int3(idx._x + 1, idx._y, idx._z - 1));

            if (idx._x > 0 && idx._y > 0 && idx._z > 0)
                output.Add(new Int3(idx._x - 1, idx._y - 1, idx._z - 1));

            if (idx._x < gridSize._x - 1 && idx._y > 0 && idx._z > 0)
                output.Add(new Int3(idx._x + 1, idx._y - 1, idx._z - 1));
            if (idx._x > 0 && idx._y < gridSize._y - 1 && idx._z > 0)
                output.Add(new Int3(idx._x - 1, idx._y + 1, idx._z - 1));
            if (idx._x > 0 && idx._y > 0 && idx._z < gridSize._z - 1)
                output.Add(new Int3(idx._x - 1, idx._y - 1, idx._z + 1));

            if (idx._x < gridSize._x - 1 && idx._y < gridSize._y - 1 && idx._z > 0)
                output.Add(new Int3(idx._x + 1, idx._y + 1, idx._z - 1));
            if (idx._x > 0 && idx._y < gridSize._y - 1 && idx._z < gridSize._z - 1)
                output.Add(new Int3(idx._x - 1, idx._y + 1, idx._z + 1));
            if (idx._x < gridSize._x - 1 && idx._y > 0 && idx._z < gridSize._z - 1)
                output.Add(new Int3(idx._x + 1, idx._y - 1, idx._z + 1));

            if (idx._x < gridSize._x - 1 && idx._y < gridSize._y - 1 && idx._z < gridSize._z - 1)
                output.Add(new Int3(idx._x + 1, idx._y + 1, idx._z + 1));

            return output;
        }

        public void FindNeighborSpace(SPHParticle particle)
        {
            particle.neighborSpace.Clear();
            Int3 idx = particle.cell.cellIdx;

            if (idx._x < 0 || idx._x >= gridSize._x ||
                idx._y < 0 || idx._y >= gridSize._y ||
                idx._z < 0 || idx._z >= gridSize._z)
                return;

            particle.neighborSpace.Add(idx);

            if (idx._x > 0)
                particle.neighborSpace.Add(new Int3(idx._x - 1, idx._y, idx._z));
            if (idx._y > 0)
                particle.neighborSpace.Add(new Int3(idx._x, idx._y - 1, idx._z));
            if (idx._z > 0)
                particle.neighborSpace.Add(new Int3(idx._x, idx._y, idx._z - 1));

            if (idx._x < gridSize._x - 1)
                particle.neighborSpace.Add(new Int3(idx._x + 1, idx._y, idx._z));
            if (idx._y < gridSize._y - 1)
                particle.neighborSpace.Add(new Int3(idx._x, idx._y + 1, idx._z));
            if (idx._z < gridSize._z - 1)
                particle.neighborSpace.Add(new Int3(idx._x, idx._y, idx._z + 1));

            if (idx._x > 0 && idx._y > 0)
                particle.neighborSpace.Add(new Int3(idx._x - 1, idx._y - 1, idx._z));
            if (idx._y > 0 && idx._z > 0)
                particle.neighborSpace.Add(new Int3(idx._x, idx._y - 1, idx._z - 1));
            if (idx._x > 0 && idx._z > 0)
                particle.neighborSpace.Add(new Int3(idx._x - 1, idx._y, idx._z - 1));

            if (idx._x < gridSize._x - 1 && idx._y < gridSize._y - 1)
                particle.neighborSpace.Add(new Int3(idx._x + 1, idx._y + 1, idx._z));
            if (idx._y < gridSize._y - 1 && idx._z < gridSize._z - 1)
                particle.neighborSpace.Add(new Int3(idx._x, idx._y + 1, idx._z + 1));
            if (idx._x < gridSize._x - 1 && idx._z < gridSize._z - 1)
                particle.neighborSpace.Add(new Int3(idx._x + 1, idx._y, idx._z + 1));

            if (idx._x > 0 && idx._y < gridSize._y - 1)
                particle.neighborSpace.Add(new Int3(idx._x - 1, idx._y + 1, idx._z));
            if (idx._y > 0 && idx._z < gridSize._z - 1)
                particle.neighborSpace.Add(new Int3(idx._x, idx._y - 1, idx._z + 1));
            if (idx._x > 0 && idx._z < gridSize._z - 1)
                particle.neighborSpace.Add(new Int3(idx._x - 1, idx._y, idx._z + 1));

            if (idx._x < gridSize._x - 1 && idx._y > 0)
                particle.neighborSpace.Add(new Int3(idx._x + 1, idx._y - 1, idx._z));
            if (idx._y < gridSize._y - 1 && idx._z > 0)
                particle.neighborSpace.Add(new Int3(idx._x, idx._y + 1, idx._z - 1));
            if (idx._x < gridSize._x - 1 && idx._z > 0)
                particle.neighborSpace.Add(new Int3(idx._x + 1, idx._y, idx._z - 1));

            if (idx._x > 0 && idx._y > 0 && idx._z > 0)
                particle.neighborSpace.Add(new Int3(idx._x - 1, idx._y - 1, idx._z - 1));

            if (idx._x < gridSize._x - 1 && idx._y > 0 && idx._z > 0)
                particle.neighborSpace.Add(new Int3(idx._x + 1, idx._y - 1, idx._z - 1));
            if (idx._x > 0 && idx._y < gridSize._y - 1 && idx._z > 0)
                particle.neighborSpace.Add(new Int3(idx._x - 1, idx._y + 1, idx._z - 1));
            if (idx._x > 0 && idx._y > 0 && idx._z < gridSize._z - 1)
                particle.neighborSpace.Add(new Int3(idx._x - 1, idx._y - 1, idx._z + 1));

            if (idx._x < gridSize._x - 1 && idx._y < gridSize._y - 1 && idx._z > 0)
                particle.neighborSpace.Add(new Int3(idx._x + 1, idx._y + 1, idx._z - 1));
            if (idx._x > 0 && idx._y < gridSize._y - 1 && idx._z < gridSize._z - 1)
                particle.neighborSpace.Add(new Int3(idx._x - 1, idx._y + 1, idx._z + 1));
            if (idx._x < gridSize._x - 1 && idx._y > 0 && idx._z < gridSize._z - 1)
                particle.neighborSpace.Add(new Int3(idx._x + 1, idx._y - 1, idx._z + 1));

            if (idx._x < gridSize._x - 1 && idx._y < gridSize._y - 1 && idx._z < gridSize._z - 1)
                particle.neighborSpace.Add(new Int3(idx._x + 1, idx._y + 1, idx._z + 1));
        }

        public bool CreateParticle(double mass, Vector3d initPos, Vector3d initVelo, out SPHParticle particle)
        {
            if (currParticleNum >= maxParticleNum || isSolving)
            {
                particle = null;
                return false;
            }

            particle = new SPHParticle();
            allParticles.Add(particle);
            particle.mass = mass;
            particle.invMass = 1 / mass;
            particle.position = initPos;
            particle.velocity = initVelo;
            particle.midVelocity = initVelo;
            particle.prevVelocity = initVelo;

            Int3 cellIdx = FindCellIdx(initPos);
            SPHGridCell cell = grid[cellIdx._x * gridCountYZ + cellIdx._y * gridSize._z + cellIdx._z];
            cell.particles.Add(particle);
            particle.cell = cell;
            return true;
        }

        public bool CreateParticle(double mass, Vector3d initPos, Vector3d initVelo)
        {
            if (currParticleNum >= maxParticleNum || isSolving)
                return false;

            SPHParticle particle = new SPHParticle();
            allParticles.Add(particle);
            particle.mass = mass;
            particle.invMass = 1 / mass;
            particle.position = initPos;
            particle.velocity = initVelo;
            particle.midVelocity = initVelo;
            particle.prevVelocity = initVelo;

            Int3 cellIdx = FindCellIdx(initPos);
            SPHGridCell cell = grid[cellIdx._x * gridCountYZ + cellIdx._y * gridSize._z + cellIdx._z];
            cell.particles.Add(particle);
            particle.cell = cell;
            return true;
        }

        public bool CreateGPUParticle(float mass, Vector3 initPos, Vector3 initVelo)
        {
            if (currParticleNum >= maxParticleNum || isSolving)
                return false;

            allCSParticles.Add(new CSParticle(mass, initPos, initVelo));

            return true;
        }

        public void UpdateParticlePressure(SPHParticle particle)
        {
            //density & pressure
            particle.density = 0;
            //List<Int3> adjacentCells = FindAdjacentCells(particle.currData.cell.cellIdx);
            for (int n = 0; n < particle.neighborSpace.Count; ++n)
            {
                SPHGridCell cell =
                    grid[particle.neighborSpace[n]._x * gridCountYZ +
                    particle.neighborSpace[n]._y * gridSize._z +
                    particle.neighborSpace[n]._z];

                foreach (SPHParticle neighbor in cell.particles)
                    particle.density += neighbor.mass * KernelPoly6(particle.position - neighbor.position);

            }

            particle.pressure = stiffness * (particle.density - restDensity);
        }

        public void UpdateParticleFluidForce(SPHParticle particle)
        {
            //force: pressure & viscosity & tension
            particle.forcePressure = Vector3d.zero;
            particle.forceViscosity = Vector3d.zero;
            particle.forceTension = Vector3d.zero;
            particle.colorGradient = Vector3d.zero;
            particle.onSurface = false;

            double tension = 0;

            for (int n = 0; n < particle.neighborSpace.Count; ++n)
            {
                SPHGridCell cell =
                    grid[particle.neighborSpace[n]._x * gridCountYZ +
                    particle.neighborSpace[n]._y * gridSize._z +
                    particle.neighborSpace[n]._z];

                foreach (SPHParticle neighbor in cell.particles)
                {
                    if (!ReferenceEquals(particle, neighbor))
                    {
                        particle.forcePressure -= 0.5f * neighbor.mass * (particle.pressure + neighbor.pressure) / neighbor.density *
                                       GradKernelSpiky(particle.position - neighbor.position);

                        particle.forceViscosity += viscosity * neighbor.mass *
                                           (neighbor.velocity - particle.velocity) / neighbor.density *
                                           LaplacianKernelViscosity(particle.position - neighbor.position);
                    }
                    particle.colorGradient += neighbor.mass / neighbor.density * GradKernelPoly6(particle.position - neighbor.position);
                    tension -= neighbor.mass / neighbor.density * LaplacianKernelPoly6(particle.position - neighbor.position);
                }
            }

            if (particle.colorGradient.sqrMagnitude > surfaceThreshold * surfaceThreshold)
            {
                particle.onSurface = true;
                particle.forceTension = tensionCoef * tension * particle.colorGradient.normalized;
            }
        }

        public void ApplyBoundaryCondition(SPHParticle particle)
        {
            //simple cube boundary
            Int3 nextIdx = FindCellIdx(particle.position);
            bool collision = false;
            Vector3d contact = particle.position;
            Vector3d contactNormal = Vector3d.zero;
            if (nextIdx._x < 0)
            {
                collision = true;
                contact.x = MathHelper.Eps;
                contactNormal.x -= 1;
            }
            if (nextIdx._x >= gridSize._x)
            {
                collision = true;
                contact.x = gridSize._x * kernelRadius - MathHelper.Eps;
                contactNormal.x += 1;
            }
            if (nextIdx._y < 0)
            {
                collision = true;
                contact.y = MathHelper.Eps;
                contactNormal.y -= 1;
            }
            if (nextIdx._y >= gridSize._y)
            {
                collision = true;
                contact.y = gridSize._y * kernelRadius - MathHelper.Eps;
                contactNormal.y += 1;
            }
            if (nextIdx._z < 0)
            {
                collision = true;
                contact.z = MathHelper.Eps;
                contactNormal.z -= 1;
            }
            if (nextIdx._z >= gridSize._z)
            {
                collision = true;
                contact.z = gridSize._z * kernelRadius - MathHelper.Eps;
                contactNormal.z += 1;
            }

            if (collision)
            {
                contactNormal.Normalize();
                Vector3d proj = Vector3d.Dot(particle.midVelocity, contactNormal) * contactNormal;
                particle.position = contact;
                particle.velocity -= (1 + 0.5) * proj;
                particle.midVelocity = 0.5 * (particle.velocity + particle.prevVelocity);
            }

        }

        public void Init()
        {
            isSolving = true;
            for (int i = 0; i < currParticleNum; ++i)
                FindNeighborSpace(allParticles[i]);
            for (int i = 0; i < currParticleNum; ++i)
                UpdateParticlePressure(allParticles[i]);
            for (int i = 0; i < currParticleNum; ++i)
            {
                SPHParticle curr = allParticles[i];
                UpdateParticleFluidForce(curr);
                Vector3d acc = curr.invMass * (curr.forcePressure + curr.forceViscosity + curr.forceTension);
                acc += externalAcc;
                curr.velocity += 0.5 * acc * timeStep;
            }
            isSolving = false;
        }

        public void Step()
        {
            isSolving = true;
            for (int i = 0; i < currParticleNum; ++i)
                FindNeighborSpace(allParticles[i]);
            for (int i = 0; i < currParticleNum; ++i)
                UpdateParticlePressure(allParticles[i]);
            for (int i = 0; i < currParticleNum; ++i)
            {
                SPHParticle curr = allParticles[i];
                UpdateParticleFluidForce(curr);
                Vector3d acc = curr.invMass * (curr.forcePressure + curr.forceViscosity + curr.forceTension);
                acc += externalAcc;
                //leap frog integration
                curr.position = curr.position + curr.velocity * timeStep;
                curr.prevVelocity = curr.velocity;
                curr.velocity = curr.velocity + acc * timeStep;
                curr.midVelocity = 0.5 * (curr.velocity + curr.prevVelocity);
                ApplyBoundaryCondition(curr);

                //cell update
                Int3 nextIdx = FindCellIdx(curr.position);

                if (!curr.cell.cellIdx.Equals(nextIdx))
                {
                    curr.cell.particles.Remove(curr);
                    curr.cell = grid[nextIdx._x * gridCountYZ + nextIdx._y * gridSize._z + nextIdx._z];
                    curr.cell.particles.Add(curr);
                }
            }

            isSolving = false;
        }

        public void SampleSurface(Vector3d pos, out double value, out Vector3d normal)
        {
            value = 0;
            normal = Vector3d.zero;
            List<Int3> neighborSpace = FindNeighborSpace(FindCellIdx(pos));
            for (int n = 0; n < neighborSpace.Count; ++n)
            {
                SPHGridCell cell = grid[neighborSpace[n]._x * gridCountYZ +
                                    neighborSpace[n]._y * gridSize._z +
                                    neighborSpace[n]._z];
                foreach (SPHParticle neighbor in cell.particles)
                {
                    value += neighbor.mass / neighbor.density * KernelPoly6(pos - neighbor.position);
                    normal += neighbor.mass / neighbor.density * GradKernelPoly6(pos - neighbor.position);
                }

            }

            if (value > 0)
            {
                value -= surfaceThreshold;
                normal.Normalize();
                normal *= -1;
            }
            else
                value = -surfaceThreshold;
        }

        public void InitOnGPU()
        {
            isSolving = true;
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

            _bufferParticleStartIndexPerCell = new ComputeBuffer(gridCountXYZ + 1, sizeof(int));
            _particleStartIndexPerCell = new int[gridCountXYZ + 1];
            _bufferParticleStartIndexPerCell.SetData(_particleStartIndexPerCell);

            _shaderSPH.SetBuffer(_kernelCci, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelCci, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelCci, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelCci, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelCci, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _bufferParticles.GetData(_allCSParticlesContainer);
            //sort based on cellIdx1d
            Array.Sort(_allCSParticlesContainer, CSParticleComparer.comparerInst);
            _bufferParticles.SetData(_allCSParticlesContainer);

            _bufferParticleNumPerCell.GetData(_particleStartIndexPerCell);
            int startIdx = 0;
            for (int i = 0; i < gridCountXYZ + 1; ++i)
            {
                int oldVal = _particleStartIndexPerCell[i];
                _particleStartIndexPerCell[i] = startIdx;
                startIdx += oldVal;
            }
            _bufferParticleStartIndexPerCell.SetData(_particleStartIndexPerCell);

            _shaderSPH.SetBuffer(_kernelFns, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelFns, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelFns, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelFns, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelFns, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _shaderSPH.SetBuffer(_kernelUpd, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelUpd, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelUpd, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelUpd, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelUpd, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _shaderSPH.SetBuffer(_kernelUpff, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelUpff, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelUpff, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelUpff, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelUpff, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _shaderSPH.SetBuffer(_kernelIp, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelIp, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelIp, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelIp, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelIp, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _bufferParticles.GetData(_allCSParticlesContainer);

            isSolving = false;
        }

        public void StepOnGPU()
        {
            isSolving = true;

            _bufferNeighborSpace.SetData(_neighborSpaceInit);
            _bufferParticleNumPerCell.SetData(_particleNumPerCellInit);

            _shaderSPH.SetBuffer(_kernelCci, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelCci, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelCci, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelCci, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelCci, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _bufferParticles.GetData(_allCSParticlesContainer);
            //sort based on cellIdx1d
            Array.Sort(_allCSParticlesContainer, CSParticleComparer.comparerInst);
            _bufferParticles.SetData(_allCSParticlesContainer);

            _bufferParticleNumPerCell.GetData(_particleStartIndexPerCell);
            int startIdx = 0;
            for (int i = 0; i < gridCountXYZ + 1; ++i)
            {
                int oldVal = _particleStartIndexPerCell[i];
                _particleStartIndexPerCell[i] = startIdx;
                startIdx += oldVal;
            }
            _bufferParticleStartIndexPerCell.SetData(_particleStartIndexPerCell);

            _shaderSPH.SetBuffer(_kernelFns, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelFns, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelFns, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelFns, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelFns, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _shaderSPH.SetBuffer(_kernelUpd, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelUpd, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelUpd, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelUpd, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelUpd, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _bufferParticles.GetData(_allCSParticlesContainer);

            _shaderSPH.SetBuffer(_kernelUpff, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelUpff, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelUpff, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelUpff, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelUpff, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _bufferParticles.GetData(_allCSParticlesContainer);

            _shaderSPH.SetBuffer(_kernelAp, "_ParticleStartIndexPerCell", _bufferParticleStartIndexPerCell);
            _shaderSPH.SetBuffer(_kernelAp, "_ParticleNumPerCell", _bufferParticleNumPerCell);
            _shaderSPH.SetBuffer(_kernelAp, "_Particles", _bufferParticles);
            _shaderSPH.SetBuffer(_kernelAp, "_NeighborSpace", _bufferNeighborSpace);

            _shaderSPH.Dispatch(_kernelAp, Mathf.CeilToInt((float)currParticleNum / 1000f), 1, 1);

            _bufferParticles.GetData(_allCSParticlesContainer);

            isSolving = false;
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

            if (_bufferParticleStartIndexPerCell != null)
            {
                _bufferParticleStartIndexPerCell.Release();
                _bufferParticleStartIndexPerCell = null;
            }
        }
    }
}
