using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace SPHFluid
{
    public struct CSParticle
    {
        public float mass;
        public float inv_density;
        public Vector3 position;
        public Vector3 velocity;
        public int onSurface;
        public Vector3 midVelocity;
        public Vector3 prevVelocity;
        public float pressure;
        public Vector3 forcePressure;
        public Vector3 forceViscosity;
        public Vector3 forceTension;
        public Vector3 colorGradient;
        public int cellIdx1d;


        public CSParticle(float mass, float inv_density, Vector3 position)
        {
            this.mass = mass;
            this.inv_density = inv_density;
            this.position = position;
            this.velocity = Vector3.zero;
            this.onSurface = 0;
            this.midVelocity = Vector3.zero;
            this.prevVelocity = Vector3.zero;
            this.pressure = 0f;
            this.forcePressure = Vector3.zero;
            this.forceViscosity = Vector3.zero;
            this.forceTension = Vector3.zero;
            this.colorGradient = Vector3.zero;
            this.cellIdx1d = 0;
        }

        public CSParticle(float mass, Vector3 position, Vector3 velocity)
        {
            this.mass = mass;
            this.inv_density = 0f;
            this.position = position;
            this.velocity = velocity;
            this.onSurface = 0;
            this.midVelocity = Vector3.zero;
            this.prevVelocity = Vector3.zero;
            this.pressure = 0f;
            this.forcePressure = Vector3.zero;
            this.forceViscosity = Vector3.zero;
            this.forceTension = Vector3.zero;
            this.colorGradient = Vector3.zero;
            this.cellIdx1d = 0;
        }

        public static int stride = sizeof(float) * 27 + sizeof(int) * 2; //NOTICE that bool is 4 bytes on GPU!
    }

    public class CSParticleComparer : IComparer
    {
        public int Compare(object x, object y)
        {
            CSParticle px = (CSParticle)x;
            CSParticle py = (CSParticle)y;
            if (px.cellIdx1d < py.cellIdx1d)
                return -1;
            else if (px.cellIdx1d > py.cellIdx1d)
                return 1;
            else return 0;
        }

        public static CSParticleComparer comparerInst = new CSParticleComparer();
    }

    public struct CSSphere
    {
        public Vector3 center;
        public float radius;
        public Vector3 velocity;
        public int active;

        public CSSphere(Vector3 center, float radius, Vector3 velocity, int active)
        {
            this.center = center;
            this.radius = radius;
            this.velocity = velocity;
            this.active = active;
        }

        public static int stride { get { return 7 * sizeof(float) + sizeof(int); } }
    }
}
