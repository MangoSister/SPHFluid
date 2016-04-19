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
        public int currParticleNum { get { return allParticles.Count; } }
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
                curr.midVelocity = curr.velocity;
                curr.prevVelocity = curr.velocity;
                curr.velocity = curr.velocity + acc * timeStep;
                curr.midVelocity = 0.5 * (curr.velocity + curr.midVelocity);
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
            //if (neighborSpace == null)
            //    Debug.Log(pos);
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
    }
}


