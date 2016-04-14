using UnityEngine;
using System.Collections;

namespace SPHFluid
{
    public class SPHSolver
    {
        public static readonly double kCommanConst = 1.566681471061;
        public static readonly double kSpikyConst = 4.774648292757;
        public static readonly double kViscosityConst = 2.387324146378;

        public static double KernelCommon(Vector3d r, double h)
        {

            return 0;
        }

        public static double KernelSpiky(Vector3d r, double h)
        {
            return 0;
        }

        public static double KernelViscosity(Vector3d r, double h)
        {
            return 0;
        }
    }
}


