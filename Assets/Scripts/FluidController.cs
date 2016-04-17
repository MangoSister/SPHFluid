using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using SPHFluid.Render;

namespace SPHFluid
{
    public class FluidController : MonoBehaviour
    {
        public MarchingCubeEngine mcEngine;
        private SPHSolver sphSolver;

        public int maxParticleNum;
        public float updateInterval;
        public double timeStep;
        public double kernelRadius;
        public double stiffness;
        public double restDensity;
        public double viscosity;
        public double tensionCoef;
        public double surfaceThreshold;

        [HideInInspector]
        public Vector3d externalAcc;
        [HideInInspector]
        public Int3 gridSize;

        private void Start()
        {
            //ExampleMc()
            sphSolver = new SPHSolver(maxParticleNum, timeStep, kernelRadius,
                                        stiffness, restDensity, externalAcc,
                                        viscosity, tensionCoef, surfaceThreshold,
                                        gridSize._x, gridSize._y, gridSize._z);


            CreateDirFlow();

            StartCoroutine(Simulate_CR());
        }

        private void OnDestroy()
        {
            mcEngine.Free();
        }

        private void CreateTest25Square()
        {
            for (int x = 0; x < 5; ++x)
                for (int z = 0; z < 5; ++z)
                {
                    sphSolver.CreateParticle(1, new Vector3d(5 + 0.1 * x, 4.9, 5 + 0.1 * z), Vector3d.zero);
                }
        }

        private void CreateDirFlow()
        {
            for (int x = 0; x < 20; ++x)
                sphSolver.CreateParticle(1, new Vector3d(5 + 0.1 * x, 4.9, 5), new Vector3d(5,0,0));
        }

        private void ExampleMc()
        {
            mcEngine.Init();
            List<Int3> list = new List<Int3>()
            {
                new Int3(0,0,0 ),
                new Int3(0,0,1 ),
                new Int3(0,1,0 ),
                new Int3(0,1,1 ),
                new Int3(1,0,0 ),
                new Int3(1,0,1 ),
                new Int3(1,1,0 ),
                new Int3(1,1,1 ),
            };

            for (int ix = 0; ix < mcEngine.width + 2; ix++)
                for (int iy = 0; iy < mcEngine.height + 2; iy++)
                    for (int iz = 0; iz < mcEngine.length + 2; iz++)
                        mcEngine.voxelSamples[ix, iy, iz] = -(new Vector3(ix, iy, iz) - Vector3.one * 8f).sqrMagnitude + 64f;

            mcEngine.BatchUpdate(list);
        }

        private IEnumerator Simulate_CR()
        {
            sphSolver.Init();
            while (true)
            {
                yield return new WaitForSeconds(updateInterval);
                //yield return null;
               // if (Input.GetKeyDown(KeyCode.S))
                    sphSolver.Step();
                //update MarchingCubeEngine
            }
        }

#if UNITY_EDITOR
        private void OnDrawGizmos()
        {
            Gizmos.color = Color.cyan;
            Vector3 extent = new Vector3(gridSize._x, gridSize._y, gridSize._z) * (float)kernelRadius * 0.5f;
            Vector3 center = transform.position + extent;
            Gizmos.DrawWireCube(center, extent * 2);
            Gizmos.color = Color.white;

            if (sphSolver != null && sphSolver.allParticles != null && sphSolver.allParticles.Count > 0)
            {
                for (int i = 0; i < sphSolver.allParticles.Count; ++i)
                {
                    Vector3 pos = transform.position;
                    pos += new Vector3((float)sphSolver.allParticles[i].currData.position.x,
                        (float)sphSolver.allParticles[i].currData.position.y,
                        (float)sphSolver.allParticles[i].currData.position.z);
                    Gizmos.DrawWireSphere(pos, 0.2f);
                }
            }
        }
#endif
    }
}