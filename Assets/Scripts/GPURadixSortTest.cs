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
    public int kernelLocalSort;
    // Use this for initialization
    void Start ()
    {
        particles = new CSParticle[512];
        for (int i = 0; i < particles.Length; ++i)
        {
            if (i < 256)
                particles[i].cellIdx1d = particles.Length - 1 - i;
            else particles[i].cellIdx1d = 3;
        }

        bufferOrdered = new ComputeBuffer(1, 4);
        bufferOrdered.SetData(new bool[1] { true });

        bufferParticles = new ComputeBuffer(particles.Length, CSParticle.stride);
        bufferParticles.SetData(particles);
        bufferSortedParticles = new ComputeBuffer(particles.Length, CSParticle.stride);

        bufferGlobalBucket = new ComputeBuffer(16, sizeof(int));
        bufferGlobalBucket.SetData(new int[16]);

        bufferLocalBucket = new ComputeBuffer(16 * (particles.Length / 256), sizeof(int));
        bufferLocalBucket.SetData(new int[16 * (particles.Length / 256)]);

        kernelLocalSort = shaderRadixSort.FindKernel("LocalSort");
    }
	
	// Update is called once per frame
	void Update ()
    {
        if (Input.GetKeyDown(KeyCode.L))
        {
            shaderRadixSort.SetInt("_ParticleNum", particles.Length);
            shaderRadixSort.SetInt("_RotationRound", 0);
            shaderRadixSort.SetBuffer(kernelLocalSort, "_Ordered", bufferOrdered);
            shaderRadixSort.SetBuffer(kernelLocalSort, "_Particles", bufferParticles);
            shaderRadixSort.SetBuffer(kernelLocalSort, "_GlobalPrefixSum", bufferGlobalBucket);
            shaderRadixSort.SetBuffer(kernelLocalSort, "_LocalPrefixSum", bufferLocalBucket);
            shaderRadixSort.SetBuffer(kernelLocalSort, "_SortedParticles", bufferSortedParticles);

            shaderRadixSort.Dispatch(kernelLocalSort, Mathf.CeilToInt((float)particles.Length / (float)256), 1, 1);

            bufferSortedParticles.GetData(particles);
            int[] gBuckets = new int[16];
            bufferGlobalBucket.GetData(gBuckets);

            int[] lBuckets = new int[16 * particles.Length / 256];
            bufferLocalBucket.GetData(lBuckets);
            //foreach (var p in particles)
            //    print(p.cellIdx1d & 15);

            //Debug.Log("wowowo");
        }
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
