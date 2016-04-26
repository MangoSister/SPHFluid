using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using SPHFluid;

public class ControlPanel : MonoBehaviour
{
    public Toggle toggle;
    public Text maxParticleNum;
    public Text textBtnRun;

    public Canvas backgroud;
    public GameObject configuration;

    public SliderValReader sliderUpdateInterval;
    public SliderValReader sliderTimeStep;
    public SliderValReader sliderKernelRadius;
    public SliderValReader sliderStiffness;
    public SliderValReader sliderRestDensityl;
    public SliderValReader sliderViscosity;
    public SliderValReader sliderTensionStrength;
    public SliderValReader sliderSurfaceThreshold;

    public SliderValReader sliderGravityX;
    public SliderValReader sliderGravityY;
    public SliderValReader sliderGravityZ;

    public SliderValReader sliderGridSizeX;
    public SliderValReader sliderGridSizeY;
    public SliderValReader sliderGridSizeZ;

    public SliderValReader sliderMeshRes;

    public FluidController ctrl;

    private bool isRunning = false;

    public void OnToggleConfiguration(bool value)
    {
        configuration.SetActive(value);
        backgroud.gameObject.SetActive(value);
    }

    public void OnRunBtnHit()
    {
        isRunning = !isRunning;
        if (isRunning)
        {
            textBtnRun.text = "Stop";
            ctrl.updateInterval = sliderUpdateInterval.GetVal();
            ctrl.timeStep = sliderTimeStep.GetVal();
            ctrl.kernelRadius = sliderKernelRadius.GetVal();
            ctrl.stiffness = sliderStiffness.GetVal();
            ctrl.restDensity = sliderRestDensityl.GetVal();
            ctrl.externalAcc.x = sliderGravityX.GetVal();
            ctrl.externalAcc.y = sliderGravityY.GetVal();
            ctrl.externalAcc.z = sliderGravityZ.GetVal();
            ctrl.gridSize._x = (int)sliderGridSizeX.GetVal();
            ctrl.gridSize._y = (int)sliderGridSizeY.GetVal();
            ctrl.gridSize._z = (int)sliderGridSizeZ.GetVal();

            ctrl.mcEngine.width = (int)sliderMeshRes.GetVal();
            ctrl.mcEngine.height = (int)sliderMeshRes.GetVal();
            ctrl.mcEngine.length = (int)sliderMeshRes.GetVal();

            toggle.isOn = false;
            ctrl.Init();
            ctrl.enabled = true;
        }
        else
        {
            textBtnRun.text = "Run";

            toggle.isOn = true;
            ctrl.enabled = false;
            ctrl.Free();
        }
    }

    private void Start()
    {
        maxParticleNum.text = string.Format("Max Particle Num: {0}", ctrl.maxParticleNum);
    }
}
