using Unity.Entities;

[InternalBufferCapacity(8)]
public struct NeighborsEntityBuffer : IBufferElementData
{
    public Entity Value;
}
