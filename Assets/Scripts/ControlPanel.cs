using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using SPHFluid;

public class ControlPanel : MonoBehaviour
{
    public Toggle toggle;
    public Text textBtnRun;

    public Canvas backgroud;
    public GameObject configuration;

    public SliderValReader sliderParticleNum;
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

    public GameObject simBorder;
    public Material matBorderLine;

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

            int particleNumLevel = (int)sliderParticleNum.GetVal() / (int)sliderParticleNum.scale;
            int meshResLevel = (int)sliderMeshRes.GetVal();
            ctrl.Init(particleNumLevel, meshResLevel);

            var uis = configuration.GetComponentsInChildren<Selectable>();
            foreach (var ui in uis)
                ui.interactable = false;

            simBorder.SetActive(true);
            simBorder.transform.localPosition = 0.5f * (float)ctrl.sphSolver.kernelRadius *  new Vector3(ctrl.sphSolver.gridSize._x, ctrl.sphSolver.gridSize._y, ctrl.sphSolver.gridSize._z);
            simBorder.transform.localScale = new Vector3(ctrl.sphSolver.gridSize._x, ctrl.sphSolver.gridSize._y, ctrl.sphSolver.gridSize._z);

            toggle.isOn = false;
            ctrl.enabled = true;
        }
        else
        {
            textBtnRun.text = "Run";
            toggle.isOn = true;
            ctrl.enabled = false;
            ctrl.Free();

            simBorder.SetActive(false);

            var uis = configuration.GetComponentsInChildren<Selectable>();
            foreach (var ui in uis)
                ui.interactable = true;
        }
    }

    //private void OnRenderObject()
    //{
    //    if (isRunning)
    //    {
    //        matBorderLine.SetPass(0);
    //        GL.PushMatrix();
    //        GL.LoadProjectionMatrix(Camera.main.projectionMatrix);
    //        GL.modelview = Camera.main.worldToCameraMatrix;
            
    //        GL.Begin(GL.LINES);
    //        GL.Color(Color.red);
    //        GL.Vertex(Vector3.zero);
    //        GL.Vertex(new Vector3((float)ctrl.sphSolver.kernelRadius * ctrl.gridSize._x, 0f, 0f));
    //        GL.End();
    //        GL.PopMatrix();
    //    }
    //}
}
