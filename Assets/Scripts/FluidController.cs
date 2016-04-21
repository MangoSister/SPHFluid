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
        public ComputeShader shaderSPH;

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
                                        gridSize._x, gridSize._y, gridSize._z, shaderSPH);


            //CreateTest25Square();
            //CreateDirFlow();
            CreateTest125Cube();
            //CreateTest1000Cube();
            //CreateTwoCollisionFlow();
            //CreatePenetrateWall();

            sphSolver.Init();

            mcEngine.engineScale = (float)gridSize._x / (float)mcEngine.width;
            mcEngine.Init(sphSolver);

            if (sphSolver.currParticleNum > 0)
                StartCoroutine(Simulate_CR());
        }

        private void OnDestroy()
        {
            sphSolver.Free();
            mcEngine.Free();
        }

        #region Test Cases
        private void CreateTest25Square()
        {
            for (int x = 0; x < 5; ++x)
                for (int z = 0; z < 5; ++z)
                {
                    sphSolver.CreateParticle(1, new Vector3d(5 + 0.1 * x, 4.9, 5 + 0.1 * z), Vector3d.zero);
                }
        }

        private void CreateTest125Cube()
        {
            for (int x = 0; x < 5; ++x)
                for (int y = 0; y < 5; ++y)
                    for (int z = 0; z < 5; ++z)
                    {
                        sphSolver.CreateParticle(1, new Vector3d(5 + 0.1 * x, 5 + 0.1 * y, 5 + 0.1 * z), Vector3d.zero);
                    }
        }

        private void CreateTest1000Cube()
        {
            for (int x = 0; x < 10; ++x)
                for (int y = 0; y < 10; ++y)
                    for (int z = 0; z < 10; ++z)
                    {
                        sphSolver.CreateParticle(1, new Vector3d(4 + 0.2 * x, 4 + 0.2 * y, 4 + 0.2 * z), Vector3d.zero);
                    }
        }


        private void CreateDirFlow()
        {
            for (int x = 0; x < 20; ++x)
                sphSolver.CreateParticle(1, new Vector3d(5 + 0.1 * x, 4.9, 5), new Vector3d(5,0,0));
        }

        private void CreateTwoCollisionFlow()
        {
            for (int x = 0; x < 50; ++x)
                sphSolver.CreateParticle(1, new Vector3d(Vector3.one * Random.value * 0.2f) + new Vector3d(3 + 0.1 * x, 5, 5), new Vector3d(3 + Random.Range(0, 2f), 0, 0));

            for (int x = 0; x < 50; ++x)
                sphSolver.CreateParticle(1, new Vector3d(Vector3.one * Random.value * 0.2f) + new Vector3d(7 - 0.1 * x, 7, 5), new Vector3d(-3 - Random.Range(0, 2f), -5, 0));
        }

        private void CreatePenetrateWall()
        {
            for (int x = 0; x < 10; ++x)
                for (int y = 0; y < 10; ++y)
                        sphSolver.CreateParticle(1, new Vector3d(4 + 0.2 * x, 4 + 0.2 * y, 5), Vector3d.zero);

            for (int z = 0; z < 10; ++z)
                sphSolver.CreateParticle(1, new Vector3d(5, 5, 1 + 0.05 * z), new Vector3d(0, 0, 5 + Random.Range(0, 2f)));
        }

        #endregion

        private IEnumerator SimulatorOnGPU_CR()
        {
            yield return null;
        }
        

        private IEnumerator Simulate_CR()
        {
            HashSet<Int3> currUpdateMCBlocks = new HashSet<Int3>(); 
            while (true)
            {
                yield return new WaitForSeconds(updateInterval);
                //yield return null;
                //if (!Input.GetKeyDown(KeyCode.S))
                //    continue;

//#if UNITY_EDITOR
//                float startTime = Time.realtimeSinceStartup;
//#endif
                sphSolver.Step();
                //update MarchingCubeEngine
                currUpdateMCBlocks.Clear();
                for (int i = 0; i < sphSolver.currParticleNum; ++i)
                {                
                    if (sphSolver.allParticles[i].onSurface)
                    {
                        Vector3d blockOffset = (sphSolver.allParticles[i].position /*- mcEngine.engineOrigin*/) / 
                                                (mcEngine.engineScale * MarchingCubeEngine.blockSize);
                        currUpdateMCBlocks.Add(Vector3d.FloorToInt3(blockOffset));
                    }
                }

                //#if UNITY_EDITOR
                //                print("Time taken: " + (Time.realtimeSinceStartup - startTime) * 1000.0f);
                //#endif
                mcEngine.BatchUpdate(new List<Int3>(currUpdateMCBlocks));
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
                    pos += new Vector3((float)sphSolver.allParticles[i].position.x,
                        (float)sphSolver.allParticles[i].position.y,
                        (float)sphSolver.allParticles[i].position.z);
                    Gizmos.DrawWireSphere(pos, 0.2f);
                }
            }
        }
#endif
    }
}