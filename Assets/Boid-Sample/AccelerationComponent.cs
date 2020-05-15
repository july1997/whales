using Unity.Entities;
using Unity.Mathematics;

[GenerateAuthoringComponent]
public struct Acceleration : IComponentData
{
    public float3 Value;
}