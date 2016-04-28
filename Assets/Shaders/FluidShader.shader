Shader "SPHFluid/Transparent Specular" 
{
    Properties 
	{
        _MainTex ("Base (RGB)", 2D) = "white" {}
        _Color ("Transparency (A only)", Color) = (0.5, 0.5, 0.5, 1)
        _Cube ("Reflection Cubemap", Cube) = "_Skybox" {}
        _ReflectColor ("Reflection Color", Color) = (1,1,1,0.5)
        _ReflectBrightness ("Reflection Brightness", Float) = 1.0
        _SpecularMap ("Specular / Reflection Map", 2D) = "white" {}
        _RimColor ("Rim Color", Color) = (0.26,0.19,0.16,0.0)
    }
    SubShader 
	{
        Tags 
		{
            "Queue"="Transparent"
            "IgnoreProjector"="True"
            "RenderType"="Transparent"
        }
		Cull Off
        CGPROGRAM

        #pragma surface surf BlinnPhong alpha
        sampler2D _MainTex;
        sampler2D _SpecularMap;
        samplerCUBE _Cube;
        fixed4 _ReflectColor;
        fixed _ReflectBrightness;
        fixed4 _Color;
        float4 _RimColor;
 
        struct Input 
		{
            float2 uv_MainTex;
            float3 worldRefl; // Used for reflection map
            float3 viewDir; // Used for rim lighting
        };
 
        // Surface shader
        void surf (Input IN, inout SurfaceOutput o) 
		{
         
            half4 c = tex2D (_MainTex, IN.uv_MainTex);
            o.Albedo = c.rgb;
            o.Alpha = _Color.a * c.a;
             
            half specular = tex2D(_SpecularMap, IN.uv_MainTex).r;
            o.Specular = specular;
             
            fixed4 reflcol = texCUBE (_Cube, IN.worldRefl);
            reflcol *= c.a;
            o.Emission = reflcol.rgb * _ReflectColor.rgb * _ReflectBrightness;
            o.Alpha = o.Alpha * _ReflectColor.a;
             
            half intensity = 1.0 - saturate(dot (normalize(IN.viewDir), o.Normal));
            o.Emission += intensity * _RimColor.rgb;
        }
        ENDCG
    } 
    FallBack "Diffuse"
}