using UnityEngine;
using System.Collections;
using SPHFluid;

public class GPURadixSortTest : MonoBehaviour
{
    public CSParticle[] particles;
    public ComputeShader shaderRadixSort;
    public ComputeBuffer bufferParticles;
    public ComputeBuffer bufferGlobalBucket;
    public ComputeBuffer bufferOrdered;
    public ComputeBuffer bufferSortedParticles;
    public ComputeBuffer bufferLocalBucket;
    public ComputeBuffer bufferGroupKeyOffset;
    public ComputeBuffer bufferLocalBinMarkers;
    public int sortSectionNum;
    public int groupNum;

    public int kernelLocalCount;
    public int kernelPrecedeScan;
    public int kernelLocalSort;
    public int kernelComputeGroupKeyOffset;
    public int kernelGlobalReorder;
    // Use this for initialization
    void Start ()
    {
        particles = new CSParticle[512];
        sortSectionNum = 64;
        groupNum = Mathf.CeilToInt((float)particles.Length / (float)sortSectionNum);
        for (int i = 0; i < particles.Length; ++i)
        {
            particles[i].cellIdx1d = particles.Length - 1 - i;
        }

        bufferOrdered = new ComputeBuffer(1, 4);
        bufferOrdered.SetData(new bool[1] { true });

        bufferParticles = new ComputeBuffer(particles.Length, CSParticle.stride);
        bufferParticles.SetData(particles);
        bufferSortedParticles = new ComputeBuffer(particles.Length, CSParticle.stride);

        bufferGlobalBucket = new ComputeBuffer(16, sizeof(int));
        bufferGlobalBucket.SetData(new int[16]);

        bufferLocalBucket = new ComputeBuffer(16 * groupNum, sizeof(int));
        bufferLocalBucket.SetData(new int[16 * groupNum]);

        bufferGroupKeyOffset = new ComputeBuffer(16 * groupNum, sizeof(int));
        bufferGroupKeyOffset.SetData(new int[16 * groupNum]);

        bufferLocalBinMarkers = new ComputeBuffer(16 * sortSectionNum * groupNum, sizeof(int));
        bufferLocalBinMarkers.SetData(new int[16 * sortSectionNum * groupNum]);

        kernelLocalCount = shaderRadixSort.FindKernel("LocalCount");
        kernelPrecedeScan = shaderRadixSort.FindKernel("PrecedeScan");
        kernelLocalSort = shaderRadixSort.FindKernel("LocalSort");
        kernelComputeGroupKeyOffset = shaderRadixSort.FindKernel("ComputeGroupKeyOffset");
        kernelGlobalReorder = shaderRadixSort.FindKernel("GlobalReorder");
    }
	
	// Update is called once per frame
	void Update ()
    {
        if (Input.GetKeyDown(KeyCode.J))
        {
            SortRound(0);
        }
        if (Input.GetKeyDown(KeyCode.K))
        {
            SortRound(1);
        }

        if (Input.GetKeyDown(KeyCode.L))
        {
            SortRound(2);
        }
    }

    private void SortRound(int round)
    {
        float t = Time.realtimeSinceStartup;
        bufferParticles.SetData(particles);
        bufferGlobalBucket.SetData(new int[16]);
        bufferLocalBucket.SetData(new int[16 * groupNum]);
        bufferGroupKeyOffset.SetData(new int[16 * groupNum]);

        shaderRadixSort.SetInt("_SortSectionNum", groupNum);
        shaderRadixSort.SetInt("_ParticleNum", particles.Length);
        shaderRadixSort.SetInt("_RotationRound", round);

        shaderRadixSort.SetBuffer(kernelLocalCount, "_Ordered", bufferOrdered);
        shaderRadixSort.SetBuffer(kernelLocalCount, "_Particles", bufferParticles);
        shaderRadixSort.SetBuffer(kernelLocalCount, "_GlobalPrefixSum", bufferGlobalBucket);
        shaderRadixSort.SetBuffer(kernelLocalCount, "_LocalPrefixSum", bufferLocalBucket);
        shaderRadixSort.SetBuffer(kernelLocalCount, "_GroupKeyOffset", bufferGroupKeyOffset);
        shaderRadixSort.SetBuffer(kernelLocalCount, "_LocalBinMarkers", bufferLocalBinMarkers);
        //shaderRadixSort.SetBuffer(kernelLocalCount, "_SortedParticles", bufferSortedParticles);

        shaderRadixSort.Dispatch(kernelLocalCount, groupNum, 1, 1);

        int[] gBuckets = new int[16];
        bufferGlobalBucket.GetData(gBuckets);

        int[] lBuckets = new int[16 * groupNum];
        bufferLocalBucket.GetData(lBuckets);

        int[] keyOffset = new int[16 * groupNum];
        bufferGroupKeyOffset.GetData(keyOffset);

        int[] binMarkers = new int[16 * sortSectionNum * groupNum];
        bufferLocalBinMarkers.GetData(binMarkers);

        shaderRadixSort.SetBuffer(kernelPrecedeScan, "_LocalBinMarkers", bufferLocalBinMarkers);
        shaderRadixSort.Dispatch(kernelPrecedeScan, groupNum, 1, 1);

        bufferLocalBinMarkers.GetData(binMarkers);

        shaderRadixSort.SetBuffer(kernelComputeGroupKeyOffset, "_GroupKeyOffset", bufferGroupKeyOffset);

        shaderRadixSort.Dispatch(kernelComputeGroupKeyOffset, 1, 1, 1);
        bufferGroupKeyOffset.GetData(keyOffset);

        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_Particles", bufferSortedParticles);
        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_SortedParticles", bufferParticles);
        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_GlobalPrefixSum", bufferGlobalBucket);
        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_LocalPrefixSum", bufferLocalBucket);
        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_GroupKeyOffset", bufferGroupKeyOffset);

        shaderRadixSort.Dispatch(kernelGlobalReorder, groupNum, 1, 1);
        bufferParticles.GetData(particles);
        //foreach (var p in particles)
        //    print((p.cellIdx1d >> (round * 4)) & 15);
        //foreach (var p in particles)
        //    print(p.cellIdx1d + " " + ((p.cellIdx1d >> (round * 4)) & 15));
        Debug.Log(string.Format("wowowo: {0}", Time.realtimeSinceStartup - t));

    }

    void OnDestroy()
    {
        if (bufferParticles != null)
        {
            bufferParticles.Release();
            bufferParticles = null;
        }

        if (bufferGlobalBucket != null)
        {
            bufferGlobalBucket.Release();
            bufferGlobalBucket = null;
        }

        if (bufferLocalBucket != null)
        {
            bufferLocalBucket.Release();
            bufferLocalBucket = null;
        }


        if (bufferGroupKeyOffset != null)
        {
            bufferGroupKeyOffset.Release();
            bufferGroupKeyOffset = null;
        }

        if (bufferOrdered != null)
        {
            bufferOrdered.Release();
            bufferOrdered = null;
        }

        if(bufferLocalBinMarkers != null)
        {
            bufferLocalBinMarkers.Release();
            bufferLocalBinMarkers = null;
        }

        if (bufferSortedParticles != null)
        {
            bufferSortedParticles.Release();
            bufferSortedParticles = null;
        }
    }
}
