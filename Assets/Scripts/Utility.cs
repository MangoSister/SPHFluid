using UnityEngine;
using System;
using System.Collections;

namespace SPHFluid
{
    [Serializable]
    public struct Vector3d
    {
        public double x, y, z;

        public Vector3d(double x = 0, double y = 0, double z = 0)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }
        public Vector3d(Vector3 vec3f)
        {
            x = vec3f.x;
            y = vec3f.y;
            z = vec3f.z;
        }

        public double inv_magnitude
        { get { return 1 / Math.Sqrt(x * x + y * y + z * z); } }

        public double magnitude
        {  get { return Math.Sqrt(x * x + y * y + z * z); } }

        public double sqrMagnitude
        { get { return x * x + y * y + z * z; } }

        public void Normalize()
        {
            double inv_mag = inv_magnitude;
            x *= inv_mag;
            y *= inv_mag;
            z *= inv_mag;
        }

        public Vector3d normalized
        {
            get
            {
                Vector3d copy = new Vector3d(x, y, z);
                copy.Normalize();
                return copy;
            }
        }

        public static Vector3d operator +(Vector3d lhs, Vector3d rhs)
        {
            return new Vector3d(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
        }

        public static Vector3d operator -(Vector3d lhs, Vector3d rhs)
        {
            return new Vector3d(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
        }

        public static Vector3d operator -(Vector3d lhs, Vector3 rhs)
        {
            return new Vector3d(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
        }

        public static Vector3d operator *(Vector3d lhs, double rhs)
        {
            return new Vector3d(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
        }

        public static Vector3d operator *(double lhs, Vector3d rhs)
        {
            return new Vector3d(lhs * rhs.x, lhs * rhs.y, lhs * rhs.z);
        }

        public static Vector3d operator /(Vector3d lhs, double rhs)
        {
            return new Vector3d(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
        }

        public static double Dot(Vector3d lhs, Vector3d rhs)
        {
            return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
        }

        public static Vector3d Lerp(Vector3d from, Vector3d to, double t)
        {
            t = Math.Min(1, Math.Max(t, 0));
            return from * (1 - t) + to * t;
        }

        public static Int3 FloorToInt3(Vector3d vec)
        {
            return new Int3((int)Math.Floor(vec.x), (int)Math.Floor(vec.y), (int)Math.Floor(vec.z));
        }

        public static readonly Vector3d zero = new Vector3d(0, 0, 0);
        public static readonly Vector3d one = new Vector3d(1, 1, 1);

        public override string ToString()
        {
            return string.Format("({0}, {1}, {2})", x, y, z);
        }
    }

    [Serializable]
    public struct Int3 : IEquatable<Int3>
    {
        public int _x, _y, _z;
        public Int3(int x, int y, int z)
        { _x = x; _y = y; _z = z; }

        public bool Equals(Int3 other)
        { return _x == other._x && _y == other._y && _z == other._z; }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            if (!(obj is Int3))
                return false;
            Int3 other = (Int3)obj;
            return _x == other._x && _y == other._y && _z == other._z;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 23 + _x.GetHashCode();
                hash = hash * 23 + _y.GetHashCode();
                hash = hash * 23 + _z.GetHashCode();
                return hash;
            }
        }

        public override string ToString()
        {
            return string.Format("({0}, {1}, {2})", _x, _y, _z);
        }
    }

    [Serializable]
    public struct Int2 : IEquatable<Int2>
    {
        public int _x, _y;
        public Int2(int x, int y)
        { _x = x; _y = y; }

        public bool Equals(Int2 other)
        { return _x == other._x && _y == other._y; }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;
            if (!(obj is Int2))
                return false;
            Int2 other = (Int2)obj;
            return _x == other._x && _y == other._y;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 23 + _x.GetHashCode();
                hash = hash * 23 + _y.GetHashCode();
                return hash;
            }
        }

        public override string ToString()
        {
            return string.Format("({0}, {1})", _x, _y);
        }
    }

    public static class MathHelper
    {
        public static readonly double Eps = 10e-5;
        public static bool InfLineIntersection(Vector2 l1p1, Vector2 l1p2, Vector2 l2p1, Vector2 l2p2, out Vector2 intersection)
        {
            intersection = Vector2.zero;
            float a1 = l1p2.y - l1p1.y;
            float b1 = l1p1.x - l1p2.x;
            float a2 = l2p2.y - l2p1.y;
            float b2 = l2p1.x - l2p2.x;
            float det = a1 * b2 - a2 * b1; //determinant (denominator)

            if (det == 0f) return false;//coincidence or parallel, two segments on the same line
            float c1 = a1 * l1p1.x + b1 * l1p1.y;
            float c2 = a2 * l2p1.x + b2 * l2p1.y;
            float det1 = c1 * b2 - c2 * b1; //determinant (numerator 2)
            float det2 = a1 * c2 - a2 * c1; //determinant (numerator 1)
            intersection.x = det1 / det;
            intersection.y = det2 / det;
            return true;
        }

        public static float TriangularInvLerp(float from, float to, float value)
        {
            var mid = 0.5f * (from + to);
            var firstHalf = Mathf.InverseLerp(from, mid, value);
            if (firstHalf <= 0f)
                return 0f;
            else if (firstHalf >= 1f)
                return 1f - Mathf.InverseLerp(mid, to, value);
            else return firstHalf;
        }
    }
}