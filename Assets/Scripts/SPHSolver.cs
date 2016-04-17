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

        public static double KernelPoly6(Vector3d r, double h)
        {
            double sqrDiff = (h * h - r.sqrMagnitude);
            if (sqrDiff < 0)
                return 0;
            double inv_h9 = 1 / (h * h * h * h * h * h * h * h * h);
            return kPoly6Const * inv_h9 * sqrDiff * sqrDiff * sqrDiff;
        }

        public static Vector3d GradKernelPoly6(Vector3d r, double h)
        {
            double sqrDiff = (h * h - r.sqrMagnitude);
            if (sqrDiff < 0)
                return Vector3d.zero;
            double inv_h9 = 1 / (h * h * h * h * h * h * h * h * h);
            return gradKPoly6Const * inv_h9 * sqrDiff * sqrDiff * r;
        }

        public static double LaplacianKernelPoly6(Vector3d r, double h)
        {
            double h2 = h * h;
            double r2 = r.sqrMagnitude;
            double sqrDiff = (h2 - r2);
            if (sqrDiff < 0)
                return 0;
            double inv_h9 = 1 / (h2 * h2 * h2 * h2 * h);
            return lapKPoly6Const * inv_h9 * sqrDiff * (3 * h2 - 7 * r2);
        }

        public static double KernelSpiky(Vector3d r, double h)
        {
            double diff = h - r.magnitude;
            if (diff < 0)
                return 0;
            double inv_h6 = 1 / (h * h * h * h * h * h);
            return kSpikyConst * inv_h6 * diff * diff * diff;
        }

        public static Vector3d GradKernelSpiky(Vector3d r, double h)
        {
            double mag = r.magnitude;
            double diff = (h - mag);
            if (diff < 0 || mag <= 0)
                return Vector3d.zero;
            double inv_h6 = 1 / (h * h * h * h * h * h);
            r /= mag;
            return gradKSpikyConst * inv_h6 * diff  * diff * r;
        }

        public static double KernelViscosity(Vector3d r, double h)
        {
            double mag = r.magnitude;
            if (h - mag < 0)
                return 0;
            double h2 = h * h;
            double inv_h3 = 1 / (h * h2);
            double sqrMag = mag * mag;
            return kViscosityConst * inv_h3 * (-0.5 * mag * sqrMag * inv_h3 + sqrMag / (h2) + 0.5 * h / mag - 1);
        }

        public static double LaplacianKernelViscosity(Vector3d r, double h)
        {
            double diff = h - r.magnitude;
            if (diff < 0)
                return 0;
            double inv_h6 = 1 / (h * h * h * h * h * h);
            return lapkViscosityConst * inv_h6 * diff;
        }
        #endregion

        public int maxParticleNum;
        public int currParticleNum { get { return allParticles.Count; } }
        public List<SPHParticle> allParticles;

        public double timeStep;
        public double kernelRadius;
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

        public SPHSolver(int maxParticleNum, double timeStep, double kernelRadius, 
                        double stiffness, double restDensity, Vector3d externalAcc, 
                        double viscosity, double tensionCoef, double surfaceThreshold,
                        int gridSizeX, int gridSizeY, int gridSizeZ)
        {
            this.maxParticleNum = maxParticleNum;
            allParticles = new List<SPHParticle>();
            allParticles.Capacity = maxParticleNum;
            this.timeStep = timeStep;
            this.kernelRadius = kernelRadius;
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
            Int3 idx = particle.currData.cell.cellIdx;

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
            particle.currData.position = initPos;
            particle.currData.velocity = initVelo;
            Int3 cellIdx = FindCellIdx(initPos);
            SPHGridCell cell = grid[cellIdx._x * gridCountYZ + cellIdx._y * gridSize._z + cellIdx._z];
            cell.particles.Add(particle);
            particle.currData.cell = cell;
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
            particle.currData.position = initPos;
            particle.currData.velocity = initVelo;
            Int3 cellIdx = FindCellIdx(initPos);
            SPHGridCell cell = grid[cellIdx._x * gridCountYZ + cellIdx._y * gridSize._z + cellIdx._z];
            cell.particles.Add(particle);
            particle.currData.cell = cell;
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
                    particle.density += neighbor.mass * KernelPoly6(particle.currData.position - neighbor.currData.position, kernelRadius);

            }

            particle.pressure = stiffness * (particle.density - restDensity);
        }

        public void UpdateParticleFluidForce(SPHParticle particle)
        {
            //force: pressure & viscosity & tension
            particle.forcePressure = Vector3d.zero;
            particle.forceViscosity = Vector3d.zero;
            particle.colorGradient = Vector3d.zero;
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
                                       GradKernelSpiky(particle.currData.position - neighbor.currData.position, kernelRadius);

                        particle.forceViscosity += viscosity * neighbor.mass *
                                           (neighbor.currData.velocity - particle.currData.velocity) / neighbor.density *
                                           LaplacianKernelViscosity(particle.currData.position - neighbor.currData.position, kernelRadius);
                    }
                    particle.colorGradient += neighbor.mass / neighbor.density * GradKernelPoly6(particle.currData.position - neighbor.currData.position, kernelRadius);
                    tension -= neighbor.mass / neighbor.density * LaplacianKernelPoly6(particle.currData.position - neighbor.currData.position, kernelRadius);
                }
            }

            if (particle.colorGradient.sqrMagnitude > surfaceThreshold * surfaceThreshold)
            {
                particle.nextData.onSurface = true;
                particle.forceTension = tensionCoef * tension * particle.colorGradient.normalized;
            }
            else
            {
                particle.nextData.onSurface = false;
                particle.forceTension = Vector3d.zero;
            }
        }

        public void ApplyBoundaryCondition(SPHParticle particle)
        {
            //simple cube boundary
            Int3 nextIdx = FindCellIdx(particle.nextData.position);
            bool collision = false;
            Vector3d contact = particle.nextData.position;
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
                double t = Math.Sqrt((contact - particle.currData.position).sqrMagnitude / 
                    (particle.nextData.position - particle.currData.position).sqrMagnitude);
                particle.nextData.position = contact;
                particle.nextData.velocity -= (1 + 0.5) * proj;
                particle.midVelocity = 0.5 * (particle.currData.velocity + particle.nextData.velocity);
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
                curr.currData.velocity += 0.5 * acc * timeStep;
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
               // if (i == 0)
                 //   Debug.Log(string.Format("{0:0.000}, {1:0.000}, {2:0.000}", curr.forcePressure.x, curr.forceViscosity.x, curr.forceTension.x));
                acc += externalAcc;
                //leap frog integration
                curr.nextData.position = curr.currData.position + curr.currData.velocity * timeStep;
                curr.nextData.velocity = curr.currData.velocity + acc * timeStep;
                curr.midVelocity = 0.5 * (curr.currData.velocity + curr.nextData.velocity);
                ApplyBoundaryCondition(curr);

                //cell update
                Int3 nextIdx = FindCellIdx(curr.nextData.position);
//#if UNITY_EDITOR
//                Debug.Log(nextIdx);
//#endif
                if (!curr.currData.cell.cellIdx.Equals(nextIdx))
                {
                    curr.nextData.cell = grid[nextIdx._x * gridCountYZ + nextIdx._y * gridSize._z + nextIdx._z];
                    curr.currData.cell.particles.Remove(curr);
                    curr.nextData.cell.particles.Add(curr);
                }
                else
                    curr.nextData.cell = curr.currData.cell;
                SPHParticleMotionInfo swap = curr.currData;
                curr.currData = curr.nextData;
                curr.nextData = swap;
            }
            isSolving = false;
        }
    }
}


