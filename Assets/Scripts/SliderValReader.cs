using UnityEngine;
using UnityEngine.UI;
using System.Collections;


public class SliderValReader : MonoBehaviour
{
    public float scale = 1;
    public Slider slider;
    public Text text;

    private void Start()
    {
        text.text = string.Format("{0:0.000}", slider.value * scale);
    }

    public void ReadVal(float value)
    {
        text.text = string.Format("{0:0.000}", value * scale);
    }

    public float GetVal()
    {
        return slider.value * scale;
    }
}
