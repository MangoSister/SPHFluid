using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace SPHFluid
{
    public class SPHParticle
    {
        public double mass;
        public double invMass;
        public Vector3d position;
        public Vector3d velocity;
        public SPHGridCell cell;
        public bool onSurface = false;
        public Vector3d midVelocity;
        public Vector3d prevVelocity;

        public List<Int3> neighborSpace;
        public double density;
        public double pressure;
        public Vector3d forcePressure;
        public Vector3d forceViscosity;
        public Vector3d forceTension;
        public Vector3d colorGradient;

        public SPHParticle()
        {
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

    public struct CSParticle
    {
        public float mass;
        public float inv_density;
        public Vector3 position;
        public Vector3 velocity;
        public bool onSurface;
        public Vector3 midVelocity;
        public Vector3 prevVelocity;
        public float pressure;
        public Vector3 forcePressure;
        public Vector3 forceViscosity;
        public Vector3 forceTension;
        public Vector3 colorGradient;

        public CSParticle(float mass, float inv_density, Vector3 position)
        {
            this.mass = mass;
            this.inv_density = inv_density;
            this.position = position;
            this.velocity = Vector3.zero;
            this.onSurface = false;
            this.midVelocity = Vector3.zero;
            this.prevVelocity = Vector3.zero;
            this.pressure = 0f;
            this.forcePressure = Vector3.zero;
            this.forceViscosity = Vector3.zero;
            this.forceTension = Vector3.zero;
            this.colorGradient = Vector3.zero;
        }

        public static int stride = sizeof(float) * 27 + 4; //NOTICE that bool is 4 bytes on GPU!
    }
}
