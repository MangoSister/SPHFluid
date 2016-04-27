using UnityEngine;
using System.Collections;

namespace SPHFluid
{
    public class GPURadixSortParticles
    {
        public ComputeShader _shaderRadixSort;
        private ComputeBuffer _bufferParticles;
        private ComputeBuffer _bufferGlobalBucket;
        private ComputeBuffer _bufferOrdered;
        private ComputeBuffer _bufferSortedParticles;
        private ComputeBuffer _bufferLocalBucket;
        private ComputeBuffer _bufferGroupKeyOffset;
        private ComputeBuffer _bufferLocalBinMarkers;

        public const int bucketSize = 16;
        public const int bucketBitNum = 4;
        public const int sortSectionNum = 64;

        private int _particleNum;
        private int _groupNum;
        private int _roundNum;

        private int _kernelCheckOrder;
        private int _kernelLocalCount;
        private int _kernelPrecedeScan;
        private int _kernelLocalSort;
        private int _kernelComputeGroupKeyOffset;
        private int _kernelGlobalReorder;

        private int[] _bufferOrderedInit;
        private int[] _bufferGlobalBucketInit;
        private int[] _bufferLocalBucketInit;
        private int[] _bufferGroupKeyOffsetInit;
        private int[] _bufferLocalBinMarkersInit;

        public GPURadixSortParticles(ComputeShader shader, int particleNum)
        {
            _shaderRadixSort = shader;

            _kernelCheckOrder = _shaderRadixSort.FindKernel("CheckOrder");
            _kernelLocalCount = _shaderRadixSort.FindKernel("LocalCount");
            _kernelPrecedeScan = _shaderRadixSort.FindKernel("PrecedeScan");
            _kernelLocalSort = _shaderRadixSort.FindKernel("LocalSort");
            _kernelComputeGroupKeyOffset = _shaderRadixSort.FindKernel("ComputeGroupKeyOffset");
            _kernelGlobalReorder = _shaderRadixSort.FindKernel("GlobalReorder");

            _particleNum = particleNum;
            _groupNum = Mathf.CeilToInt((float)_particleNum / (float)sortSectionNum);

            for (_roundNum = 1; _roundNum < 7; ++_roundNum)
            {
                if ((1 << (4 * _roundNum)) > _particleNum)
                    break;
            }

            _bufferOrderedInit = new int[1] { 1 };
            _bufferGlobalBucketInit = new int[bucketSize];
            _bufferLocalBucketInit = new int[bucketSize * _groupNum];
            _bufferGroupKeyOffsetInit = new int[bucketSize * _groupNum];
            _bufferLocalBinMarkersInit = new int[bucketSize * sortSectionNum * _groupNum];

            _bufferSortedParticles = new ComputeBuffer(_particleNum, CSParticle.stride);

            _bufferOrdered = new ComputeBuffer(1, sizeof(int));
            _bufferGlobalBucket = new ComputeBuffer(bucketSize, sizeof(int));
            _bufferLocalBucket = new ComputeBuffer(bucketSize * _groupNum, sizeof(int));
            _bufferGroupKeyOffset = new ComputeBuffer(bucketSize * _groupNum, sizeof(int));
            _bufferLocalBinMarkers = new ComputeBuffer(bucketSize * sortSectionNum * _groupNum, sizeof(int));
        }

        public void Init(ComputeBuffer bufferParticles)
        {
            //for (int i = 0; i < _particles.Length; ++i)
            //    _particles[i].cellIdx1d = _particles.Length - 1 - i;

            _bufferOrdered.SetData(_bufferOrderedInit);
            _bufferParticles = bufferParticles;            
            _bufferGlobalBucket.SetData(_bufferGlobalBucketInit);         
            _bufferLocalBucket.SetData(_bufferLocalBucketInit);        
            _bufferGroupKeyOffset.SetData(_bufferGroupKeyOffsetInit);
            _bufferLocalBinMarkers.SetData(_bufferLocalBinMarkersInit);
        }

        //private void Update()
        //{
        //    if (Input.GetKeyDown(KeyCode.J))
        //    {
        //        SortRound(0);
        //    }

        //    if (Input.GetKeyDown(KeyCode.U))
        //    {
        //        Debug.Log(CheckOrder());
        //    }

        //    if (Input.GetKeyDown(KeyCode.K))
        //    {
        //        SortRound(1);
        //    }

        //    if (Input.GetKeyDown(KeyCode.L))
        //    {
        //        SortRound(2);
        //    }

        //    if (Input.GetKeyDown(KeyCode.N))
        //    {
        //        SortRound(3);
        //    }
        //}

        public bool CheckOrder()
        {
            _shaderRadixSort.SetInt("_ParticleNum", _particleNum);

            int[] ordered = new int[1] { 1 };
            _bufferOrdered.SetData(ordered);

            _shaderRadixSort.SetBuffer(_kernelCheckOrder, "_Ordered", _bufferOrdered);
            _shaderRadixSort.SetBuffer(_kernelCheckOrder, "_Particles", _bufferParticles);

            _shaderRadixSort.Dispatch(_kernelCheckOrder, _groupNum, 1, 1);

            _bufferOrdered.GetData(ordered);
            return ordered[0] == 1;
        }

        public void SortRound(int round)
        {
            //float t = Time.realtimeSinceStartup;

            _bufferGlobalBucket.SetData(new int[bucketSize]);
            _bufferLocalBucket.SetData(new int[bucketSize * _groupNum]);
            _bufferGroupKeyOffset.SetData(new int[bucketSize * _groupNum]);

            _shaderRadixSort.SetInt("_SortSectionNum", _groupNum);
            _shaderRadixSort.SetInt("_ParticleNum", _particleNum);
            _shaderRadixSort.SetInt("_RotationRound", round);

            _shaderRadixSort.SetBuffer(_kernelLocalCount, "_Ordered", _bufferOrdered);
            _shaderRadixSort.SetBuffer(_kernelLocalCount, "_Particles", _bufferParticles);
            _shaderRadixSort.SetBuffer(_kernelLocalCount, "_GlobalPrefixSum", _bufferGlobalBucket);
            _shaderRadixSort.SetBuffer(_kernelLocalCount, "_LocalPrefixSum", _bufferLocalBucket);
            _shaderRadixSort.SetBuffer(_kernelLocalCount, "_GroupKeyOffset", _bufferGroupKeyOffset);
            _shaderRadixSort.SetBuffer(_kernelLocalCount, "_LocalBinMarkers", _bufferLocalBinMarkers);

            _shaderRadixSort.Dispatch(_kernelLocalCount, _groupNum, 1, 1);

            //int[] gBuckets = new int[bucketSize];
            //bufferGlobalBucket.GetData(gBuckets);

            //int[] lBuckets = new int[bucketSize * groupNum];
            //bufferLocalBucket.GetData(lBuckets);

            //int[] keyOffset = new int[bucketSize * groupNum];
            //bufferGroupKeyOffset.GetData(keyOffset);

            //int[] binMarkers = new int[bucketSize * sortSectionNum * groupNum];
            //bufferLocalBinMarkers.GetData(binMarkers);

            _shaderRadixSort.SetBuffer(_kernelPrecedeScan, "_LocalBinMarkers", _bufferLocalBinMarkers);
            _shaderRadixSort.Dispatch(_kernelPrecedeScan, _groupNum, 1, 1);

            //bufferLocalBinMarkers.GetData(binMarkers);

            _shaderRadixSort.SetBuffer(_kernelLocalSort, "_Particles", _bufferParticles);
            _shaderRadixSort.SetBuffer(_kernelLocalSort, "_LocalPrefixSum", _bufferLocalBucket);
            _shaderRadixSort.SetBuffer(_kernelLocalSort, "_LocalBinMarkers", _bufferLocalBinMarkers);
            _shaderRadixSort.SetBuffer(_kernelLocalSort, "_SortedParticles", _bufferSortedParticles);
            _shaderRadixSort.Dispatch(_kernelLocalSort, _groupNum, 1, 1);

            _shaderRadixSort.SetBuffer(_kernelComputeGroupKeyOffset, "_GroupKeyOffset", _bufferGroupKeyOffset);
            _shaderRadixSort.Dispatch(_kernelComputeGroupKeyOffset, 1, 1, 1);
            //bufferGroupKeyOffset.GetData(keyOffset);

            _shaderRadixSort.SetBuffer(_kernelGlobalReorder, "_Particles", _bufferSortedParticles);
            _shaderRadixSort.SetBuffer(_kernelGlobalReorder, "_SortedParticles", _bufferParticles);
            _shaderRadixSort.SetBuffer(_kernelGlobalReorder, "_GlobalPrefixSum", _bufferGlobalBucket);
            _shaderRadixSort.SetBuffer(_kernelGlobalReorder, "_LocalPrefixSum", _bufferLocalBucket);
            _shaderRadixSort.SetBuffer(_kernelGlobalReorder, "_GroupKeyOffset", _bufferGroupKeyOffset);

            _shaderRadixSort.Dispatch(_kernelGlobalReorder, _groupNum, 1, 1);
            //_bufferParticles.GetData(_particles);

            //foreach (var p in particles)
            //    print((p.cellIdx1d >> (round * 4)) & 15);
            //foreach (var p in _particles)
            //    print(p.cellIdx1d + " " + ((p.cellIdx1d >> (round * 4)) & 15));
            //Debug.Log(string.Format("wowowo: {0}", Time.realtimeSinceStartup - t));

        }

        public void Sort()
        {
            for (int i = 0; i < _roundNum; ++i)
            {
                //if (CheckOrder())
                //    break;
                SortRound(i);
            }
        }

        public void Free()
        {
            //if (_bufferParticles != null)
            //{
            //    _bufferParticles.Release();
            //    _bufferParticles = null;
            //}

            if (_bufferGlobalBucket != null)
            {
                _bufferGlobalBucket.Release();
                _bufferGlobalBucket = null;
            }

            if (_bufferLocalBucket != null)
            {
                _bufferLocalBucket.Release();
                _bufferLocalBucket = null;
            }


            if (_bufferGroupKeyOffset != null)
            {
                _bufferGroupKeyOffset.Release();
                _bufferGroupKeyOffset = null;
            }

            if (_bufferOrdered != null)
            {
                _bufferOrdered.Release();
                _bufferOrdered = null;
            }

            if (_bufferLocalBinMarkers != null)
            {
                _bufferLocalBinMarkers.Release();
                _bufferLocalBinMarkers = null;
            }

            if (_bufferSortedParticles != null)
            {
                _bufferSortedParticles.Release();
                _bufferSortedParticles = null;
            }
        }
    }
}