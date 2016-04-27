using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using SPHFluid.Render;

namespace SPHFluid
{
    public class FluidController : MonoBehaviour
    {
        public MarchingCubeEngine mcEngine;
        public SPHSolver sphSolver;
        public ComputeShader shaderSPH;
        public ComputeShader shaderRadixSort;
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

        private float _timer;

        

        public void Init(int particleNumLevel, int meshResLevel)
        {
            //ExampleMc()
            sphSolver = new SPHSolver(maxParticleNum, timeStep, kernelRadius,
                                        stiffness, restDensity, externalAcc,
                                        viscosity, tensionCoef, surfaceThreshold,
                                        gridSize._x, gridSize._y, gridSize._z, shaderSPH,
                                        shaderRadixSort);

            SpawnParticleBlock(particleNumLevel);
            //CreateTest125CubeGPU();
            //CreateTest1000CubeGPU();
            //CreateTest10000CubeGPU();
            //sphSolver.Init();
            sphSolver.InitOnGPU();

            mcEngine.width = sphSolver.gridSize._x * meshResLevel;
            mcEngine.height = sphSolver.gridSize._y * meshResLevel;
            mcEngine.length = sphSolver.gridSize._z * meshResLevel;

            mcEngine.engineScale = (float)kernelRadius / (float)meshResLevel;
            mcEngine.Init(sphSolver);

            _timer = 0f;
        }

        public void Free()
        {
            if (sphSolver != null)
            {
                sphSolver.Free();
            }
            if (mcEngine != null)
                mcEngine.Free();
            System.GC.Collect();
        }

        private void OnDestroy()
        {
            Free();
        }

        #region Test Cases
        private void CreateTest125CubeGPU()
        {
            for (int x = 0; x < 5; ++x)
                for (int y = 0; y < 5; ++y)
                    for (int z = 0; z < 5; ++z)
                    {
                        sphSolver.CreateGPUParticle(1, new Vector3(5f + 0.1f * x, 5f + 0.1f * y, 5f + 0.1f * z), Vector3.zero);
                    }
        }

        private void CreateTest1000CubeGPU()
        {
            for (int x = 0; x < 10; ++x)
                for (int y = 0; y < 10; ++y)
                    for (int z = 0; z < 10; ++z)
                    {
                        sphSolver.CreateGPUParticle(1, new Vector3(4f + 0.2f * x, 4f + 0.2f * y, 4f + 0.2f * z), Vector3.zero);
                    }
        }

        private void CreateTest10000CubeGPU()
        {
            for (int x = 0; x < 20; ++x)
                for (int y = 0; y < 25; ++y)
                    for (int z = 0; z < 20; ++z)
                    {
                        sphSolver.CreateGPUParticle(1, new Vector3(0.01f + 0.49f * x, 0.01f + 0.39f * y, 0.01f + 0.49f * z), Vector3.zero);
                    }
        }

        private readonly static Int3[] exampleBlockDim = new Int3[10]
            {
                new Int3(10, 10, 10), new Int3(10, 20, 10),
                new Int3(15, 20, 10), new Int3(20, 20, 10),
                new Int3(20, 25, 10), new Int3(20, 30, 10),
                new Int3(20, 35, 10), new Int3(20, 20, 20),
                new Int3(20, 25, 18), new Int3(20, 25, 20),
            };
        private void SpawnParticleBlock(int level)
        {
            if (level <= 0 || level > 10)
                return;
            Int3 dim = exampleBlockDim[level - 1];
            Vector3 center = 0.5f * (float)kernelRadius * new Vector3(sphSolver.gridSize._x, sphSolver.gridSize._y, sphSolver.gridSize._z);
            Vector3 extent = 0.5f * (float)kernelRadius * new Vector3(sphSolver.gridSize._x - 2, sphSolver.gridSize._y - 2, sphSolver.gridSize._z - 2);
            Vector3 interval = Vector3.Scale(2 * extent, new Vector3(1f / (dim._x - 1), 1f / (dim._y - 1), 1f / (dim._z - 1)));
            for (int x = 0; x < dim._x; ++x)
                for (int y = 0; y < dim._y; ++y)
                    for (int z = 0; z < dim._z; ++z)
                        sphSolver.CreateGPUParticle(1, Random.insideUnitSphere * 0.01f + center - extent + new Vector3(x * interval.x, y * interval.y, z * interval.z), Vector3.zero);

        }
        #endregion

        private void Update()
        {
            _timer += Time.deltaTime;
            if (_timer > updateInterval)
            {
                _timer -= updateInterval;
                sphSolver.StepOnGPU();
                mcEngine.BatchUpdate();
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

            //if (sphSolver != null && sphSolver._allCSParticlesContainer != null && sphSolver._allCSParticlesContainer.Length > 0)
            //{
            //    for (int i = 0; i < sphSolver._allCSParticlesContainer.Length; ++i)
            //    {
            //        Vector3 pos = transform.position;
            //        pos += new Vector3((float)sphSolver._allCSParticlesContainer[i].position.x,
            //            (float)sphSolver._allCSParticlesContainer[i].position.y,
            //            (float)sphSolver._allCSParticlesContainer[i].position.z);
            //        Gizmos.DrawWireSphere(pos, 0.2f);
            //    }
            //}
        }
#endif
    }
}