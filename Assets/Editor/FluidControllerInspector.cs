#if UNITY_EDITOR

using UnityEngine;
using UnityEditor;
using System.Collections;

namespace SPHFluid
{
    [CustomEditor(typeof(FluidController))]
    public class FluidControllerInspector : Editor
    {
        private FluidController ctrl { get { return target as FluidController; } }
        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();
            serializedObject.Update();
            ShowVector3d(ref ctrl.externalAcc, "External Acc (x,y,z)");
            ShowInt3(ref ctrl.gridSize, "Grid Size (x,y,z)");
            serializedObject.ApplyModifiedProperties();
        }

        private void ShowVector3d(ref Vector3d vec, string title)
        {
            EditorGUILayout.LabelField(title);
            EditorGUILayout.BeginHorizontal();
            vec.x = EditorGUILayout.FloatField((float)vec.x);
            vec.y = EditorGUILayout.FloatField((float)vec.y);
            vec.z = EditorGUILayout.FloatField((float)vec.z);
            EditorGUILayout.EndHorizontal();
        }

        private void ShowInt3(ref Int3 vec, string title)
        {
            EditorGUILayout.LabelField(title);
            EditorGUILayout.BeginHorizontal();
            vec._x = EditorGUILayout.IntField(vec._x);
            vec._y = EditorGUILayout.IntField(vec._y);
            vec._z = EditorGUILayout.IntField(vec._z);
            EditorGUILayout.EndHorizontal();
        }
    }


}

#endif