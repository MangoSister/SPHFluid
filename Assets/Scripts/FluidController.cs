using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using SPHFluid.Render;

namespace SPHFluid
{
    public class FluidController : MonoBehaviour
    {
        public MarchingCubeEngine mcEngine;

        private void Start()
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
                        mcEngine.voxelSamples[ix, iy, iz] = - (new Vector3(ix, iy, iz) - Vector3.one * 8f).sqrMagnitude + 64f;

            mcEngine.BatchUpdate(list);
        }

        private void OnDestroy()
        {
            mcEngine.Free();
        }
    }
}