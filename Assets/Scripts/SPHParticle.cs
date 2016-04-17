using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace SPHFluid
{
    public class SPHParticleMotionInfo
    {
        public Vector3d position;
        public Vector3d velocity;
        public SPHGridCell cell;
        public bool onSurface = false;
    }


    public class SPHParticle
    {
        public SPHParticleMotionInfo currData;
        public SPHParticleMotionInfo nextData;
        public double mass;
        public double invMass;
        public Vector3d midVelocity;

        public List<Int3> neighborSpace;
        public double density;
        public double pressure;
        public Vector3d forcePressure;
        public Vector3d forceViscosity;
        public Vector3d forceTension;
        public Vector3d colorGradient;

        public SPHParticle()
        {
            currData = new SPHParticleMotionInfo();
            nextData = new SPHParticleMotionInfo();
            neighborSpace = new List<Int3>();
            neighborSpace.Capacity = 27;
        }
    }

    public class SPHGridCell
    {
        public HashSet<SPHParticle> particles;
        public Int3 cellIdx;

        public SPHGridCell(int x, int y, int z)
        {
            particles = new HashSet<SPHParticle>();
            cellIdx = new Int3(x, y, z);
        }
    }
}
