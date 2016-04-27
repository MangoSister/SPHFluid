using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
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

    public GameObject funOption;
    public GameObject ball0, ball1;

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

            Vector3 simCenter = 0.5f * (float)ctrl.sphSolver.kernelRadius * new Vector3(ctrl.sphSolver.gridSize._x, ctrl.sphSolver.gridSize._y, ctrl.sphSolver.gridSize._z);

            simBorder.SetActive(true);
            simBorder.transform.localPosition = simCenter;
            simBorder.transform.localScale = new Vector3(ctrl.sphSolver.gridSize._x, ctrl.sphSolver.gridSize._y, ctrl.sphSolver.gridSize._z) * (float)ctrl.sphSolver.kernelRadius;

            toggle.isOn = false;

            float radius = 1f;
            Vector3 velo = Vector3.zero;
            ctrl.sphSolver._obstacles = new List<CSSphere>()
                                            { new CSSphere(simCenter, radius, velo, 0),
                                              new CSSphere(simCenter, radius, velo, 0) };
            ball0.transform.position = simCenter;
            ball0.transform.localScale = Vector3.one * radius;
            ball0.transform.rotation = Quaternion.identity;
            ball1.transform.position = simCenter;
            ball1.transform.localScale = Vector3.one * radius;
            ball1.transform.rotation = Quaternion.identity;
            ctrl.enabled = true;

            funOption.gameObject.SetActive(true);
            var toggles = funOption.GetComponentsInChildren<Toggle>();
            foreach (var tog in toggles)
                tog.isOn = false;

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

            funOption.gameObject.SetActive(false);
            var toggles = funOption.GetComponentsInChildren<Toggle>();
            foreach (var tog in toggles)
                tog.isOn = false;
            ball0.SetActive(false);
            ball1.SetActive(false);
        }
    }

    public void OnToggleObs1(bool value)
    {
        ball0.SetActive(value);
        var s0 = ctrl.sphSolver._obstacles[0];
        s0.active = System.Convert.ToInt32(value);
        ctrl.sphSolver._obstacles[0] = s0;
    }

    public void OnToggleObs2(bool value)
    {
        ball1.SetActive(value);
        var s1 = ctrl.sphSolver._obstacles[1];
        s1.active = System.Convert.ToInt32(value);
        ctrl.sphSolver._obstacles[1] = s1;
    }

    private void FixedUpdate()
    {
        if (!isRunning)
            return;
        if (ctrl.sphSolver._obstacles[0].active != 0)
        {
            CSSphere s0 = ctrl.sphSolver._obstacles[0];
            s0.velocity = Vector3.zero;
            if (Input.GetKey(KeyCode.W))
            {
                s0.center.z += 0.1f;
                s0.velocity.z += 0.1f / Time.fixedDeltaTime;
            }
            else if (Input.GetKey(KeyCode.S))
            {
                s0.center.z -= 0.1f;
                s0.velocity.z -= 0.1f / Time.fixedDeltaTime;
            }
            if (Input.GetKey(KeyCode.A))
            {
                s0.center.x -= 0.1f;
                s0.velocity.x -= 0.1f / Time.fixedDeltaTime;
            }
            else if (Input.GetKey(KeyCode.D))
            {
                s0.center.x += 0.1f;
                s0.velocity.x += 0.1f / Time.fixedDeltaTime;
            }
            if (Input.GetKey(KeyCode.Q))
            {
                s0.center.y += 0.1f;
                s0.velocity.y += 0.1f / Time.fixedDeltaTime;
            }
            else if (Input.GetKey(KeyCode.E))
            {
                s0.center.y -= 0.1f;
                s0.velocity.y -= 0.1f / Time.fixedDeltaTime;
            }

            ctrl.sphSolver._obstacles[0] = s0;
            ball0.transform.position = s0.center;
        }

        if (ctrl.sphSolver._obstacles[1].active != 0)
        {
            CSSphere s1 = ctrl.sphSolver._obstacles[1];
            s1.velocity = Vector3.zero;

            if (Input.GetKey(KeyCode.I))
            {
                s1.center.z += 0.1f;
                s1.velocity.z += 0.1f / Time.fixedDeltaTime;
            }
            else if (Input.GetKey(KeyCode.K))
            {
                s1.center.z -= 0.1f;
                s1.velocity.z -= 0.1f / Time.fixedDeltaTime;
            }
            if (Input.GetKey(KeyCode.J))
            {
                s1.center.x -= 0.1f;
                s1.velocity.x -= 0.1f / Time.fixedDeltaTime;
            }
            else if (Input.GetKey(KeyCode.L))
            {
                s1.center.x += 0.1f;
                s1.velocity.x += 0.1f / Time.fixedDeltaTime;
            }
            if (Input.GetKey(KeyCode.U))
            {
                s1.center.y += 0.1f;
                s1.velocity.y += 0.1f / Time.fixedDeltaTime;
            }
            else if (Input.GetKey(KeyCode.O))
            {
                s1.center.y -= 0.1f;
                s1.velocity.y -= 0.1f / Time.fixedDeltaTime;
            }

            ctrl.sphSolver._obstacles[1] = s1;
            ball1.transform.position = s1.center;
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
