using UnityEngine;
using System;
using System.Linq;
using System.Collections;
using System.Collections.Generic;

namespace SPHFluid.Render
{
    /// <summary>
    /// The main class of voxel terrain mesh generation:
    /// * divide the whole world space into blocks along 3 dimensions (basic chunk system)
    /// * maintain a queue to store terrain modifiers and apply them in update routines
    /// * sample density function in the modified area, send data to GPU
    /// * dispatch GPU-based Marching Cubes to generate mesh, read data back from GPU
    /// * apply materials and collider to mesh
    /// * other maintainence jobs
    /// </summary>
    public class MarchingCubeEngine : MonoBehaviour
    {
        /// <summary>
        /// The struct used to read triangle data back from GPU
        /// </summary>
        private struct CSTriangle
        {
            // triangle vertices positons
            public Vector3 position0;
            public Vector3 position1;
            public Vector3 position2;
            // normals of vertices
            public Vector3 normal0;
            public Vector3 normal1;
            public Vector3 normal2;
            // indicates which block it belongs to
            public int block;
            // how many memory does one CSTriangle takes? (we have 6 Vector3 and one int)
            public static int stride = sizeof(float) * 3 * 6 + sizeof(int);
        };

        //x: width, y: height, z: length
        public int width = 16, height = 16, length = 16; // voxel size
        public int blockDimX { get { return width / blockSize; } }
        public int blockDimY { get { return height / blockSize; } }
        public int blockDimZ { get { return length / blockSize; } }
        // The samples of the terrain density function
        //public float[, ,] voxelSamples;
        //public delegate void ImplicitSurface(int x, int y, int z, out float value, out Vector3 normal); //(x,y,z) : voxel index
        //public ImplicitSurface implicitSurface;

        // The size of _voxelSamples should not exceed [1025,1025,1025]
        public const int maxSampleResolution = 1025; //sample size = voxel size + 1   

        //Since we only care about the zero isosurface of the density function,
        //the density function is clamped. However, making all clamped value the same will
        //make the definition of normal vector meaningless. (0,0,0). So here we just clamp any value
        //out of bound to some random value
        public const float voidDensity = -1f;

        //each block consists of 8x8x8 voxels
        public const int blockSize = 8; // voxel size = sample size - 1
        //In order to also recompute the normals of vertices, we need to also sample "one more layer" in adjacent cells
        //"argumented block size"
        public const int ag1BlockSize = blockSize + 1;

        //The world space is divided into blocks. Each block hold its mesh.
        private GameObject[, ,] _blocks;

        //Compute shaders that should be plugged in.
        public ComputeShader shaderSample;
        public ComputeShader shaderCollectTriNum;
        public ComputeShader shaderMarchingCube;

        private int sampleKernel;
        private int ctnKernel;
        private int mcKernel;
        private int _kernelCollectSurfaceBlock;

        //Mesh material that should be plugged in
        public Material material;

        //Constant Marching Cubes compute buffer. They only need to be loaded once at the very beginning
        private ComputeBuffer _bufferCornerToEdgeTable;
        private ComputeBuffer _bufferCornerToTriNumTable;
        private ComputeBuffer _bufferCornerToVertTable;
        //private ComputeBuffer _bufferParticles;
        //private ComputeBuffer _bufferParticlesStartIndex;

        //The position of engine (engineTransform.position -> origin)
        public Transform engineTransform;

        public Vector3 engineOrigin { get { return engineTransform.position; } set { engineTransform.position = value; } }

        //The actual size of the engine (scaled by engineScale) .
        public Vector3 engineDim { get { return new Vector3(width, height, length) * engineScale; } }

        //The size of a voxel cell (units)
        public float engineScale = 1f;
        //voxel index -> world pos: (voxel index - 0) * engineScale + engineOrigin
        //world pos -> voxel index (space): (world pos - engineOrigin) / engineScale

        public SPHSolver sphSolver;
        private ComputeBuffer _bufferSufaceBlocks;
        private int[] _surfaceBlocksContainer;
        private int[] _surfaceBlocksInit;
        private List<Int3> _nextUpdateblocks;
        private static int[] _incSequence = Enumerable.Range(0, 65535).ToArray();

#if UNITY_EDITOR
        public bool drawBlockWireFrames = false;
        public bool drawCellWireFrames = false;
#endif
        /// <summary>
        /// The initialization. 
        /// Allcate block game objects
        /// Allocate compute buffers
        /// Initial other data sturctures, parameters, etc
        /// </summary>
        public void Init(SPHSolver sphSolver)
        {
            if (material == null)
                throw new UnityException("null material");

            if (shaderSample == null)
                throw new UnityException("null shader: _shaderSample");

            sampleKernel = shaderSample.FindKernel("SampleFluid");
            if (sampleKernel < 0)
                throw new UnityException("Fail to find kernel of shader: " + shaderSample.name);

            if (shaderCollectTriNum == null)
                throw new UnityException("null shader: _shaderCollectTriNum");

            ctnKernel = shaderCollectTriNum.FindKernel("CollectTriNum");
            if (ctnKernel < 0)
                throw new UnityException("Fail to find kernel of shader: " + shaderCollectTriNum.name);

            if (shaderMarchingCube == null)
                throw new UnityException("null shader: _shaderMarchingCube");

            mcKernel = shaderMarchingCube.FindKernel("MarchingCube");
            if (mcKernel < 0)
                throw new UnityException("Fail to find kernel of shader: " + shaderMarchingCube.name);

            if (width % blockSize != 0 || height % blockSize != 0 || length % blockSize != 0)
                throw new UnityException("block size must align to grid size");

            if (width + 1 > maxSampleResolution || height + 1 > maxSampleResolution || length + 1 > maxSampleResolution)
                throw new UnityException("too high resolution (exceeds " + maxSampleResolution + ")");

            _blocks = new GameObject[width / blockSize, height / blockSize, length / blockSize];

            _bufferCornerToEdgeTable = new ComputeBuffer(256, sizeof(int));
            _bufferCornerToEdgeTable.SetData(cornerToEdgeTable);
            _bufferCornerToTriNumTable = new ComputeBuffer(256, sizeof(int));
            _bufferCornerToTriNumTable.SetData(cornerToTriNumTable);
            _bufferCornerToVertTable = new ComputeBuffer(256 * 15, sizeof(int));
            _bufferCornerToVertTable.SetData(cornerToVertTable);

            for (int x = 0; x < width / blockSize; x++)
                for (int y = 0; y < height / blockSize; y++)
                    for (int z = 0; z < length / blockSize; z++)
                    {
                        _blocks[x, y, z] = new GameObject();
                        _blocks[x, y, z].name = "block (" + x.ToString() + "," + y.ToString() + "," + z.ToString() + ")";
                        _blocks[x, y, z].AddComponent<MeshFilter>();
                        _blocks[x, y, z].AddComponent<MeshRenderer>();
                        //_blocks[x, y, z].AddComponent<MeshCollider>();

                        _blocks[x, y, z].transform.parent = this.engineTransform;
                        var pos = new Vector3(x, y, z) * (float)blockSize * engineScale;
                        _blocks[x, y, z].transform.localPosition = pos;
                    }

            this.sphSolver = sphSolver;

            _bufferSufaceBlocks = new ComputeBuffer((width * height * length) / (blockSize * blockSize * blockSize), sizeof(int));
            _surfaceBlocksInit = new int[(width * height * length) / (blockSize * blockSize * blockSize)];
            _surfaceBlocksContainer = new int[(width * height * length) / (blockSize * blockSize * blockSize)];
            _nextUpdateblocks = new List<Int3>();
            _bufferSufaceBlocks.SetData(_surfaceBlocksInit);
            _kernelCollectSurfaceBlock = shaderSample.FindKernel("CollectSurfaceBlock");

            //shaderSample.SetInts("_SphGridSize", sphSolver.gridSize._x, sphSolver.gridSize._y, sphSolver.gridSize._z);
            //shaderSample.SetFloat("_KernelRadius", (float)sphSolver.kernelRadius);
            //shaderSample.SetInt("_ParticleNum", sphSolver.currParticleNum);
            //shaderSample.SetFloat("_inv_KernelRadius", 1f / (float)sphSolver.kernelRadius);
            //shaderSample.SetFloat("_kr2", (float)sphSolver.kr2);
            //shaderSample.SetFloat("_inv_kr3", (float)sphSolver.inv_kr3);
            //shaderSample.SetFloat("_inv_kr6", (float)sphSolver.inv_kr6);
            //shaderSample.SetFloat("_inv_kr9", (float)sphSolver.inv_kr9);
            //shaderSample.SetFloat("_SurfaceThreshold", (float)sphSolver.surfaceThreshold);
            //shaderSample.SetInts("_MCEngineDim", blockDimX, blockDimY, blockDimZ);
            //shaderSample.SetFloat("_MCVoxelScale", engineScale);

            shaderSample.SetFloat("_TimeStep", (float)sphSolver.timeStep);
            shaderSample.SetInts("_SphGridSize", sphSolver.gridSize._x, sphSolver.gridSize._y, sphSolver.gridSize._z);
            shaderSample.SetFloat("_KernelRadius", (float)sphSolver.kernelRadius);
            shaderSample.SetFloat("_inv_KernelRadius", 1f / (float)sphSolver.kernelRadius);
            shaderSample.SetFloat("_kr2", (float)sphSolver.kr2);
            shaderSample.SetFloat("_inv_kr3", (float)sphSolver.inv_kr3);
            shaderSample.SetFloat("_inv_kr6", (float)sphSolver.inv_kr6);
            shaderSample.SetFloat("_inv_kr9", (float)sphSolver.inv_kr9);
            shaderSample.SetFloat("_Stiffness", (float)sphSolver.stiffness);
            shaderSample.SetFloat("_RestDensity", (float)sphSolver.restDensity);
            shaderSample.SetFloats("_ExternalAcc", (float)sphSolver.externalAcc.x, (float)sphSolver.externalAcc.y, (float)sphSolver.externalAcc.z);
            shaderSample.SetFloat("_Viscosity", (float)sphSolver.viscosity);
            shaderSample.SetFloat("_TensionCoef", (float)sphSolver.tensionCoef);
            shaderSample.SetFloat("_SurfaceThreshold", (float)sphSolver.surfaceThreshold);
            shaderSample.SetFloat("_Eps", (float)MathHelper.Eps);
            shaderSample.SetInts("_MCEngineDim", blockDimX, blockDimY, blockDimZ);
            shaderSample.SetFloat("_MCVoxelScale", engineScale);
            shaderSample.SetInt("_ParticleNum", sphSolver.currParticleNum);
        }

        /// <summary>
        /// Free all meshes, and constant compute buffers
        /// </summary>
        public void Free()
        {
            if (_blocks != null)
            {
                for (int x = 0; x < width / blockSize; x++)
                    for (int y = 0; y < height / blockSize; y++)
                        for (int z = 0; z < length / blockSize; z++)
                        {
                            if (_blocks[x, y, z] == null)
                                continue;
                            if (_blocks[x, y, z].GetComponent<MeshFilter>().mesh != null)
                                _blocks[x, y, z].GetComponent<MeshFilter>().mesh.Clear(false);
                            //if (_blocks[x, y, z].GetComponent<MeshCollider>().sharedMesh != null)
                             //   _blocks[x, y, z].GetComponent<MeshCollider>().sharedMesh.Clear();
                            Destroy(_blocks[x, y, z]);
                            _blocks[x, y, z] = null;
                        }
            }

            if (_bufferCornerToEdgeTable != null)
            {
                _bufferCornerToEdgeTable.Release();
                _bufferCornerToEdgeTable = null;
            }

            if(_bufferCornerToTriNumTable != null)
            {
                _bufferCornerToTriNumTable.Release();
                _bufferCornerToTriNumTable = null;
            }

            if (_bufferCornerToVertTable != null)
            {
                _bufferCornerToVertTable.Release();
                _bufferCornerToVertTable = null;
            }

            if(_bufferSufaceBlocks != null)
            {
                _bufferSufaceBlocks.Release();
                _bufferSufaceBlocks = null;
            }

            //if (_bufferParticles != null)
            //{
            //    _bufferParticles.Release();
            //    _bufferParticles = null;
            //}

            //if(_bufferParticlesStartIndex != null)
            //{
            //    _bufferParticlesStartIndex.Release();
            //    _bufferParticlesStartIndex = null;
            //}
        }   

        /// <summary>
        /// Batch update. Reconstruct all meshes needed by running GPU based Marching Cubes
        /// </summary>
        public void BatchUpdate()
        { 
            for (int x = 0; x < width / blockSize; x++)
                for (int y = 0; y < height / blockSize; y++)
                    for (int z = 0; z < length / blockSize; z++)
                    {
                        if (_blocks[x, y, z] == null)
                            continue;
                        if (_blocks[x, y, z].GetComponent<MeshFilter>().mesh != null)
                            _blocks[x, y, z].GetComponent<MeshFilter>().mesh.Clear();
                        //if (_blocks[x, y, z].GetComponent<MeshCollider>().sharedMesh != null)
                        //    _blocks[x, y, z].GetComponent<MeshCollider>().sharedMesh.Clear();
                    }

            shaderSample.SetBuffer(_kernelCollectSurfaceBlock, "_Particles", sphSolver._bufferParticles);
            _bufferSufaceBlocks.SetData(_surfaceBlocksInit);
            shaderSample.SetBuffer(_kernelCollectSurfaceBlock, "_SurfaceMCBlocks", _bufferSufaceBlocks);
            shaderSample.Dispatch(_kernelCollectSurfaceBlock, sphSolver._sphthreadGroupNum, 1, 1);
            _bufferSufaceBlocks.GetData(_surfaceBlocksContainer);
            _nextUpdateblocks.Clear();

            for (int i = 0; i < _surfaceBlocksContainer.Length; ++i)
            {
                if (_surfaceBlocksContainer[i] == 1)
                {
                    int x = i / (blockDimY * blockDimZ);
                    int y = (i - x * (blockDimY * blockDimZ)) / blockDimZ;
                    int z = i - x * (blockDimY * blockDimZ) - y * blockDimZ;
                    _nextUpdateblocks.Add(new Int3(x, y, z));
                }
            }

            //for (int x = 0; x < blockDimX; ++x)
            //    for (int y = 0; y < blockDimY; ++y)
            //        for (int z = 0; z < blockDimZ; ++z)
            //            _nextUpdateblocks.Add(new Int3(x, y, z));

            if (_nextUpdateblocks.Count == 0)
                return;

            //GPU sampling
            ComputeBuffer bufferBlocks = new ComputeBuffer(_nextUpdateblocks.Count, sizeof(int) * 3);
            bufferBlocks.SetData(_nextUpdateblocks.ToArray());

            
            //List<CSParticle> particlesCopy = new List<CSParticle>();            
            
            //int[] particlesStartIdx = new int[sphSolver.gridCountXYZ + 1];

            //int startIdx = 0;
            //for (int x = 0; x < sphSolver.gridSize._x; ++x)
            //    for (int y = 0; y < sphSolver.gridSize._y; ++y)
            //        for (int z = 0; z < sphSolver.gridSize._z; ++z)
            //        {
            //            int idx = x * sphSolver.gridCountYZ + y * sphSolver.gridSize._z + z;
            //            foreach (var particle in sphSolver.grid[idx].particles)
            //            {
            //                particlesCopy.Add(new CSParticle((float)particle.mass, 1f / (float)particle.density,
            //                    new Vector3((float)particle.position.x, (float)particle.position.y, (float)particle.position.z)));
            //            }
            //            particlesStartIdx[idx] = startIdx;
            //            startIdx += sphSolver.grid[idx].particles.Count;
            //        }
            //particlesStartIdx[particlesStartIdx.Length - 1] = sphSolver.currParticleNum;

            //_bufferParticlesStartIndex.SetData(particlesStartIdx);
            //_bufferParticles.SetData(particlesCopy.ToArray());

            ComputeBuffer bufferSamples = new ComputeBuffer(_nextUpdateblocks.Count * (ag1BlockSize) * (ag1BlockSize) * (ag1BlockSize), sizeof(float));
            ComputeBuffer bufferNormals = new ComputeBuffer(_nextUpdateblocks.Count * (ag1BlockSize) * (ag1BlockSize) * (ag1BlockSize), sizeof(float) * 3);

            shaderSample.SetBuffer(sampleKernel, "_Blocks", bufferBlocks);
            shaderSample.SetBuffer(sampleKernel, "_ParticleCellNumPrefixSum", sphSolver._bufferParticleNumPerCell);
            shaderSample.SetBuffer(sampleKernel, "_Particles", sphSolver._bufferParticles);
            shaderSample.SetBuffer(sampleKernel, "_Samples", bufferSamples);
            shaderSample.SetBuffer(sampleKernel, "_Normals", bufferNormals);

            shaderSample.Dispatch(sampleKernel, _nextUpdateblocks.Count, 1, 1);

            bufferBlocks.Release();
            bufferBlocks = null;

            //marching-cube

            //STAGE I: collect triangle number
            ComputeBuffer bufferTriNum = new ComputeBuffer(1, sizeof(int));
            bufferTriNum.SetData(new int[] { 0 });
            ComputeBuffer bufferCornerFlags = new ComputeBuffer(_nextUpdateblocks.Count * blockSize * blockSize * blockSize, sizeof(int));
            
            shaderCollectTriNum.SetBuffer(ctnKernel, "_Samples", bufferSamples);
            shaderCollectTriNum.SetBuffer(ctnKernel, "_CornerToTriNumTable", _bufferCornerToTriNumTable);
            shaderCollectTriNum.SetBuffer(ctnKernel, "_TriNum", bufferTriNum);
            shaderCollectTriNum.SetBuffer(ctnKernel, "_CornerFlags", bufferCornerFlags);
            
            shaderCollectTriNum.Dispatch(ctnKernel, _nextUpdateblocks.Count, 1, 1);
            
            int[] triNum = new int[1];
            bufferTriNum.GetData(triNum);
            if(triNum[0] == 0)
            {
                //no triangles, early exit
                
                bufferNormals.Release();
                bufferSamples.Release();

                bufferTriNum.Release();
                bufferCornerFlags.Release();
                return;
            }
//#if UNITY_EDITOR
//            //Debug.Log("triangles count " + triNum[0]);
//#endif
            //STAGE II: do marching cube
            ComputeBuffer bufferMeshes = new ComputeBuffer(triNum[0], CSTriangle.stride);
            ComputeBuffer bufferTriEndIndex = new ComputeBuffer(1, sizeof(int));
            bufferTriEndIndex.SetData(new int[] { 0 });
            ComputeBuffer bufferTriBlockNum = new ComputeBuffer(_nextUpdateblocks.Count, sizeof(int));
            int[] triBlockNum = new int[_nextUpdateblocks.Count];
            bufferTriBlockNum.SetData(triBlockNum);

            shaderMarchingCube.SetBuffer(mcKernel, "_Samples", bufferSamples);
            shaderMarchingCube.SetBuffer(mcKernel, "_Normals", bufferNormals);
            shaderMarchingCube.SetBuffer(mcKernel, "_CornerFlags", bufferCornerFlags);
            shaderMarchingCube.SetBuffer(mcKernel, "_CornerToEdgeTable", _bufferCornerToEdgeTable);
            shaderMarchingCube.SetBuffer(mcKernel, "_CornerToVertTable", _bufferCornerToVertTable);
            shaderMarchingCube.SetBuffer(mcKernel, "_Meshes", bufferMeshes);
            shaderMarchingCube.SetBuffer(mcKernel, "_TriEndIndex", bufferTriEndIndex);
            shaderMarchingCube.SetBuffer(mcKernel, "_TriBlockNum", bufferTriBlockNum);
            //dispatch compute shader
            shaderMarchingCube.Dispatch(mcKernel, _nextUpdateblocks.Count, 1, 1);
       
            //split bufferMeshes to meshes for individual blocks
            CSTriangle[] csTriangles = new CSTriangle[triNum[0]];//triNum[0] is the counter
            bufferMeshes.GetData(csTriangles);
            bufferTriBlockNum.GetData(triBlockNum);
            int[] triBlockCounter = new int[triBlockNum.Length];

            Vector3[][] vertices = new Vector3[_nextUpdateblocks.Count][];
            Vector3[][] normals = new Vector3[_nextUpdateblocks.Count][];
            for (int i = 0; i < _nextUpdateblocks.Count; i++)
            {
                vertices[i] = new Vector3[3 * triBlockNum[i]];
                normals[i] = new Vector3[3 * triBlockNum[i]];
            }
            foreach (var vt in csTriangles)
            {
                vertices[vt.block][triBlockCounter[vt.block]] = vt.position0 * engineScale;
                normals[vt.block][triBlockCounter[vt.block]] = vt.normal0;
                triBlockCounter[vt.block]++;

                vertices[vt.block][triBlockCounter[vt.block]] = vt.position1 * engineScale;
                normals[vt.block][triBlockCounter[vt.block]] = vt.normal1;
                triBlockCounter[vt.block]++;

                vertices[vt.block][triBlockCounter[vt.block]] = vt.position2 * engineScale;
                normals[vt.block][triBlockCounter[vt.block]] = vt.normal2;
                triBlockCounter[vt.block]++;
            }

            for (int i = 0; i < _nextUpdateblocks.Count; i++)
            {
                var x = _nextUpdateblocks[i]._x;
                var y = _nextUpdateblocks[i]._y;
                var z = _nextUpdateblocks[i]._z;                
                _blocks[x, y, z].GetComponent<MeshFilter>().mesh.Clear(false);
                var mesh = _blocks[x, y, z].GetComponent<MeshFilter>().mesh;

                mesh.vertices = vertices[i];
                int[] idx = new int[vertices[i].Length];
                Buffer.BlockCopy(_incSequence, 0, idx, 0, vertices[i].Length * sizeof(int));
                mesh.SetTriangles(idx, 0);
                mesh.normals = normals[i];
                mesh.Optimize();
                mesh.RecalculateBounds();
               // _blocks[x, y, z].GetComponent<MeshFilter>().mesh = mesh;
                _blocks[x, y, z].GetComponent<MeshRenderer>().material = material;
                //_blocks[x, y, z].GetComponent<MeshCollider>().sharedMesh = mesh;
            }

            //Debug.Log("Time taken: " + (Time.realtimeSinceStartup - startTime) * 1000.0f);
//#if UNITY_EDITOR
//            print("Time taken: " + (Time.realtimeSinceStartup - startTime) * 1000.0f);
//#endif
            bufferNormals.Release();
            bufferSamples.Release();

            bufferTriNum.Release();
            bufferCornerFlags.Release();

            bufferMeshes.Release();
            bufferTriEndIndex.Release();
            bufferTriBlockNum.Release();

            GC.Collect();
        }
#if UNITY_EDITOR
        private void OnDrawGizmos()
        {
            if (_blocks == null || !drawBlockWireFrames)
                return;
            
            for (int x = 0; x < width / blockSize; x++)
                for (int y = 0; y < height / blockSize; y++)
                    for (int z = 0; z < length / blockSize; z++)
                    {
                        Gizmos.color = Color.green;
                        Vector3 blockOrigin =
                            engineOrigin + new Vector3(x, y, z) * blockSize * engineScale;
                        Gizmos.DrawWireCube(blockOrigin + Vector3.one * (blockSize / 2) * engineScale, Vector3.one * blockSize * engineScale);

                        if (!drawCellWireFrames)
                            continue;

                        Gizmos.color = Color.red;
                        for (int ix = 0; ix < blockSize; ix++)
                            for (int iy = 0; iy < blockSize; iy++)
                                for (int iz = 0; iz < blockSize; iz++)
                                {
                                    Vector3 cellOrigin = blockOrigin + new Vector3(ix, iy, iz) * engineScale;
                                    Gizmos.DrawWireCube(cellOrigin + Vector3.one * 0.5f * engineScale, Vector3.one * engineScale);
                                }
                    }
           

            Gizmos.color = Color.white;
        }
#endif
        /*****      data for marching-cube      *****/
        public const int maxTriNumPerCell = 5;
     
        // three looking up tables

        //cornerToEdgeTable[256]
        //Map every case to all the edges that include boundary points.
        //Since there are 12 edges in a block, 12 bits are required. 
        //These bits are packed together in an integer.    
        //This table is originally appeared on http://scrawkblog.com/2014/10/16/marching-cubes-on-the-gpu-in-unity/
        static int[] cornerToEdgeTable = new int[]
	    {
		    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 
		    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 
		    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 
		    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 
		    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60, 
		    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 
		    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 
		    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 
		    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0, 
		    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650, 
		    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 
		    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460, 
		    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0, 
		    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230, 
		    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190, 
		    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
	    };

        //cornerToTriNum[256]
        //Map every case to the number of triangles that will be generated. 
        static int[] cornerToTriNumTable = new int[] 
        {
	        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,2,
	        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,3,
	        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,3,
	        2,3,3,2,3,4,4,3,3,4,4,3,4,5,5,2,
	        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,3,
	        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,4,
	        2,3,3,4,3,4,2,3,3,4,4,5,4,5,3,2,
	        3,4,4,3,4,5,3,2,4,5,5,4,5,2,4,1,
	        1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,3,
	        2,3,3,4,3,4,4,5,3,2,4,3,4,3,5,2,
	        2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,4,
	        3,4,4,3,4,5,5,4,4,3,5,2,5,4,2,1,
	        2,3,3,4,3,4,4,5,3,4,4,5,2,3,3,2,
	        3,4,4,5,4,5,5,2,4,3,5,4,3,2,4,1,
	        3,4,4,5,4,5,3,4,4,5,5,2,3,4,2,1,
	        2,3,3,2,3,4,2,1,3,2,4,1,2,1,1,0
        };

        //cornerToVertTable[256][15]
        //The original lookup table. 
        static int[,] cornerToVertTable = new int[,] 
        {
            {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1},
            {3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1},
            {3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1},
            {3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1},
            {9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1},
            {1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1},
            {9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1},
            {2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1},
            {8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1},
            {9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1},
            {4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1},
            {3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1},
            {1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1},
            {4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1},
            {4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1},
            {9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1},
            {1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1},
            {5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1},
            {2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1},
            {9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1},
            {0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1},
            {2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1},
            {10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1},
            {4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1},
            {5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1},
            {5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1},
            {9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1},
            {0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1},
            {1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1},
            {10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1},
            {8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1},
            {2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1},
            {7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1},
            {9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1},
            {2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1},
            {11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1},
            {9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1},
            {5,7,0,5,0,9,7,11,0,1,0,10,11,10,0},
            {11,10,0,11,0,3,10,5,0,8,0,7,5,7,0},
            {11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1},
            {1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1},
            {9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1},
            {5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1},
            {2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1},
            {0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1},
            {5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1},
            {6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1},
            {0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1},
            {3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1},
            {6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1},
            {5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1},
            {1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1},
            {10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1},
            {6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1},
            {1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1},
            {8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1},
            {7,3,9,7,9,4,3,2,9,5,9,6,2,6,9},
            {3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1},
            {5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1},
            {0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1},
            {9,2,1,9,11,2,9,4,11,7,11,4,5,10,6},
            {8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1},
            {5,1,11,5,11,6,1,0,11,7,11,4,0,4,11},
            {0,5,9,0,6,5,0,3,6,11,6,3,8,4,7},
            {6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1},
            {10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1},
            {10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1},
            {8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1},
            {1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1},
            {3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1},
            {0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1},
            {10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1},
            {0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1},
            {3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1},
            {6,4,1,6,1,10,4,8,1,2,1,11,8,11,1},
            {9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1},
            {8,11,1,8,1,0,11,6,1,9,1,4,6,4,1},
            {3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1},
            {6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1},
            {0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1},
            {10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1},
            {10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1},
            {1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1},
            {2,6,9,2,9,1,6,7,9,0,9,3,7,3,9},
            {7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1},
            {7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1},
            {2,0,7,2,7,11,0,9,7,6,7,10,9,10,7},
            {1,8,0,1,7,8,1,10,7,6,7,10,2,3,11},
            {11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1},
            {8,9,6,8,6,7,9,1,6,11,6,3,1,3,6},
            {0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1},
            {7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1},
            {10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1},
            {2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1},
            {6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1},
            {7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1},
            {2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1},
            {1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1},
            {10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1},
            {10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1},
            {0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1},
            {7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1},
            {6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1},
            {8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1},
            {9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1},
            {6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1},
            {1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1},
            {4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1},
            {10,9,3,10,3,2,9,4,3,11,3,6,4,6,3},
            {8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1},
            {0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1},
            {1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1},
            {8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1},
            {10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1},
            {4,6,3,4,3,8,6,10,3,0,3,9,10,9,3},
            {10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1},
            {5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1},
            {11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1},
            {9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1},
            {6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1},
            {7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1},
            {3,4,8,3,5,4,3,2,5,10,5,2,11,7,6},
            {7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1},
            {9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1},
            {3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1},
            {6,2,8,6,8,7,2,1,8,4,8,5,1,5,8},
            {9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1},
            {1,6,10,1,7,6,1,0,7,8,7,0,9,5,4},
            {4,0,10,4,10,5,0,3,10,6,10,7,3,7,10},
            {7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1},
            {6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1},
            {3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1},
            {0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1},
            {6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1},
            {1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1},
            {0,11,3,0,6,11,0,9,6,5,6,9,1,2,10},
            {11,8,5,11,5,6,8,0,5,10,5,2,0,2,5},
            {6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1},
            {5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1},
            {9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1},
            {1,5,8,1,8,0,5,6,8,3,8,2,6,2,8},
            {1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,3,6,1,6,10,3,8,6,5,6,9,8,9,6},
            {10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1},
            {0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1},
            {5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1},
            {10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1},
            {11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1},
            {0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1},
            {9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1},
            {7,5,2,7,2,11,5,9,2,3,2,8,9,8,2},
            {2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1},
            {8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1},
            {9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1},
            {9,8,2,9,2,1,8,7,2,10,2,5,7,5,2},
            {1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1},
            {9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1},
            {9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1},
            {5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1},
            {0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1},
            {10,11,4,10,4,5,11,3,4,9,4,1,3,1,4},
            {2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1},
            {0,4,11,0,11,3,4,5,11,2,11,1,5,1,11},
            {0,2,5,0,5,9,2,11,5,4,5,8,11,8,5},
            {9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1},
            {5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1},
            {3,10,2,3,5,10,3,8,5,4,5,8,0,1,9},
            {5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1},
            {8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1},
            {0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1},
            {9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1},
            {0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1},
            {1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1},
            {3,1,4,3,4,8,1,10,4,7,4,11,10,11,4},
            {4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1},
            {9,7,4,9,11,7,9,1,11,2,11,1,0,8,3},
            {11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1},
            {11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1},
            {2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1},
            {9,10,7,9,7,4,10,2,7,8,7,0,2,0,7},
            {3,7,10,3,10,2,7,4,10,1,10,0,4,0,10},
            {1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1},
            {4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1},
            {4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1},
            {0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1},
            {3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1},
            {3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1},
            {0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1},
            {9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1},
            {1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
            {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
        };
    }
}

