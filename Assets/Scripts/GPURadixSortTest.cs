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

    public int sortSectionNum;

    public int kernelLocalSort;
    public int kernelComputeGroupKeyOffset;
    public int kernelGlobalReorder;
    // Use this for initialization
    void Start ()
    {
        particles = new CSParticle[512];
        sortSectionNum = 64;
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

        bufferLocalBucket = new ComputeBuffer(16 * (particles.Length / sortSectionNum), sizeof(int));
        bufferLocalBucket.SetData(new int[16 * (particles.Length / sortSectionNum)]);

        bufferGroupKeyOffset = new ComputeBuffer(16 * (particles.Length / sortSectionNum), sizeof(int));
        bufferGroupKeyOffset.SetData(new int[16 * (particles.Length / sortSectionNum)]);

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
        bufferLocalBucket.SetData(new int[16 * (particles.Length / sortSectionNum)]);
        bufferGroupKeyOffset.SetData(new int[16 * (particles.Length / sortSectionNum)]);

        shaderRadixSort.SetInt("_SortSectionNum", Mathf.CeilToInt((float)particles.Length / (float)sortSectionNum));
        shaderRadixSort.SetInt("_ParticleNum", particles.Length);
        shaderRadixSort.SetInt("_RotationRound", round);
        shaderRadixSort.SetBuffer(kernelLocalSort, "_Ordered", bufferOrdered);
        shaderRadixSort.SetBuffer(kernelLocalSort, "_Particles", bufferParticles);
        shaderRadixSort.SetBuffer(kernelLocalSort, "_GlobalPrefixSum", bufferGlobalBucket);
        shaderRadixSort.SetBuffer(kernelLocalSort, "_LocalPrefixSum", bufferLocalBucket);
        shaderRadixSort.SetBuffer(kernelLocalSort, "_GroupKeyOffset", bufferGroupKeyOffset);
        shaderRadixSort.SetBuffer(kernelLocalSort, "_SortedParticles", bufferSortedParticles);

        shaderRadixSort.Dispatch(kernelLocalSort, Mathf.CeilToInt((float)particles.Length / (float)sortSectionNum), 1, 1);

        bufferSortedParticles.GetData(particles);
        int[] gBuckets = new int[16];
        bufferGlobalBucket.GetData(gBuckets);

        int[] lBuckets = new int[16 * particles.Length / sortSectionNum];
        bufferLocalBucket.GetData(lBuckets);

        int[] keyOffset = new int[16 * particles.Length / sortSectionNum];
        bufferGroupKeyOffset.GetData(keyOffset);
        //foreach (var p in particles)
        //    print(p.cellIdx1d & 15);

        shaderRadixSort.SetBuffer(kernelComputeGroupKeyOffset, "_GroupKeyOffset", bufferGroupKeyOffset);
        shaderRadixSort.Dispatch(kernelComputeGroupKeyOffset, 1, 1, 1);


        bufferGroupKeyOffset.GetData(keyOffset);

        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_Particles", bufferSortedParticles);
        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_SortedParticles", bufferParticles);
        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_GlobalPrefixSum", bufferGlobalBucket);
        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_LocalPrefixSum", bufferLocalBucket);
        shaderRadixSort.SetBuffer(kernelGlobalReorder, "_GroupKeyOffset", bufferGroupKeyOffset);

        shaderRadixSort.Dispatch(kernelGlobalReorder, Mathf.CeilToInt((float)particles.Length / (float)sortSectionNum), 1, 1);
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

        if (bufferSortedParticles != null)
        {
            bufferSortedParticles.Release();
            bufferSortedParticles = null;
        }
    }
}
