using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using Unity.Physics;

using UnityEngine;

public class BoidsSimulationSystem : JobComponentSystem
{
    EntityQuery group;
    
    protected override void OnCreate()
    {
        group = GetEntityQuery(typeof(Translation), typeof(Velocity), typeof(NeighborsEntityBuffer));
    }
        
    [BurstCompile]
    public struct NeighborsDetectionJob : IJobForEachWithEntity<Translation, Velocity>
    {
        [ReadOnly] public float prodThresh;
        [ReadOnly] public float distThresh;
        [ReadOnly] public ComponentDataFromEntity<Translation> positionFromEntity;
        [NativeDisableParallelForRestriction] public BufferFromEntity<NeighborsEntityBuffer> neighborsFromEntity;
        [DeallocateOnJobCompletion] [ReadOnly] public NativeArray<Entity> entities;

        public void Execute(
            Entity entity,
            int index,
            [ReadOnly] ref Translation pos,
            [ReadOnly] ref Velocity velocity)
        {
            neighborsFromEntity[entity].Clear();

            float3 pos0 = pos.Value;
            float3 fwd0 = math.normalize(velocity.Value);

            for (int i = 0; i < entities.Length; ++i)
            {
                var neighbor = entities[i];
                if (neighbor == entity) continue;

                float3 pos1 = positionFromEntity[neighbor].Value;
                var to = pos1 - pos0;
                var dist = math.length(to);

                if (dist < distThresh)
                {
                    var dir = math.normalize(to);
                    var prod = Unity.Mathematics.math.dot(dir, fwd0);
                    if (prod > prodThresh)
                    {
                        neighborsFromEntity[entity].Add(new NeighborsEntityBuffer { Value = neighbor });
                    }
                }
            }
        }
    }

    [BurstCompile]
    private struct WallJob : IJobForEach<Translation, Acceleration>
    {
        [ReadOnly] public float scale;
        [ReadOnly] public float thresh;
        [ReadOnly] public float weight;

        public void Execute([ReadOnly] ref Translation pos, ref Acceleration accel)
        {
            accel = new Acceleration
            {
                Value = accel.Value +
                    GetAccelAgainstWall(-scale - pos.Value.x, new float3(+1, 0, 0), thresh, weight) +
                    GetAccelAgainstWall(-scale - pos.Value.y, new float3(0, +1, 0), thresh, weight) +
                    GetAccelAgainstWall(-scale - pos.Value.z, new float3(0, 0, +1), thresh, weight) +
                    GetAccelAgainstWall(+scale - pos.Value.x, new float3(-1, 0, 0), thresh, weight) +
                    GetAccelAgainstWall(+scale - pos.Value.y, new float3(0, -1, 0), thresh, weight) +
                    GetAccelAgainstWall(+scale - pos.Value.z, new float3(0, 0, -1), thresh, weight)
            };
        }

        float3 GetAccelAgainstWall(float dist, float3 dir, float thresh, float weight)
        {
            if (dist < thresh)
            {
                return dir * (weight / math.abs(dist / thresh));
            }
            return float3.zero;
        }
    }

    [BurstCompile]
    public struct SeparationJob : IJobForEachWithEntity<Translation, Acceleration>
    {
        [ReadOnly] public float separationWeight;
        [ReadOnly] public BufferFromEntity<NeighborsEntityBuffer> neighborsFromEntity;
        [ReadOnly] public ComponentDataFromEntity<Translation> positionFromEntity;

        public void Execute(Entity entity, int index, [ReadOnly] ref Translation pos, ref Acceleration accel)
        {
            var neighbors = neighborsFromEntity[entity].Reinterpret<Entity>();
            if (neighbors.Length == 0) return;

            var pos0 = pos.Value;

            var force = float3.zero;
            for (int i = 0; i < neighbors.Length; ++i)
            {
                var pos1 = positionFromEntity[neighbors[i]].Value;
                force += math.normalize(pos0 - pos1);
            }
            force /= neighbors.Length;

            var dAccel = force * separationWeight;
            accel = new Acceleration { Value = accel.Value + dAccel };
        }
    }

    [BurstCompile]
    public struct AlignmentJob : IJobForEachWithEntity<Velocity, Acceleration>
    {
        [ReadOnly] public float alignmentWeight;
        [ReadOnly] public BufferFromEntity<NeighborsEntityBuffer> neighborsFromEntity;
        [ReadOnly] public ComponentDataFromEntity<Velocity> velocityFromEntity;

        public void Execute(Entity entity, int index, [ReadOnly] ref Velocity velocity, ref Acceleration accel)
        {
            var neighbors = neighborsFromEntity[entity].Reinterpret<Entity>();
            if (neighbors.Length == 0) return;

            var averageVelocity = float3.zero;
            for (int i = 0; i < neighbors.Length; ++i)
            {
                averageVelocity += velocityFromEntity[neighbors[i]].Value;
            }
            averageVelocity /= neighbors.Length;

            var dAccel = (averageVelocity - velocity.Value) * alignmentWeight;
            accel = new Acceleration { Value = accel.Value + dAccel };
        }
    }

    [BurstCompile]
    public struct CohesionJob : IJobForEachWithEntity<Translation, Acceleration>
    {
        [ReadOnly] public float cohesionWeight;
        [ReadOnly] public BufferFromEntity<NeighborsEntityBuffer> neighborsFromEntity;
        [ReadOnly] public ComponentDataFromEntity<Translation> positionFromEntity;

        public void Execute(Entity entity, int index, [ReadOnly] ref Translation pos, ref Acceleration accel)
        {
            var neighbors = neighborsFromEntity[entity].Reinterpret<Entity>();
            if (neighbors.Length == 0) return;

            var averagePos = float3.zero;
            for (int i = 0; i < neighbors.Length; ++i)
            {
                averagePos += positionFromEntity[neighbors[i]].Value;
            }
            averagePos /= neighbors.Length;

            var dAccel = (averagePos - pos.Value) * cohesionWeight;
            accel = new Acceleration { Value = accel.Value + dAccel };
        }
    }

    [BurstCompile]
    private struct MoveJob : IJobForEach<Translation, Rotation, Velocity, Acceleration>
    {
        [ReadOnly] public float dt;
        [ReadOnly] public float minSpeed;
        [ReadOnly] public float maxSpeed;

        public void Execute(ref Translation translation, ref Rotation rot, ref Velocity velocity, ref Acceleration accel)
        {
            var v = velocity.Value;
            v += accel.Value * dt;
            var dir = math.normalize(v);
            var speed = math.length(v);
            v = math.clamp(speed, minSpeed, maxSpeed) * dir;

            translation = new Translation { Value = translation.Value + v * dt };
            rot = new Rotation { Value = quaternion.LookRotationSafe(dir, new float3(0, 1, 0)) };
            velocity = new Velocity { Value = v };
            accel = new Acceleration { Value = float3.zero };
        }
    }

    protected override JobHandle OnUpdate(JobHandle inputDeps)
    {
         var neighbors = new NeighborsDetectionJob
        {
            prodThresh = math.cos(math.radians(Bootstrap.Param.neighborFov)),
            distThresh = Bootstrap.Param.neighborDistance,
            neighborsFromEntity = GetBufferFromEntity<NeighborsEntityBuffer>(false),
            positionFromEntity = GetComponentDataFromEntity<Translation>(true),
            entities = group.ToEntityArray(Allocator.TempJob),
        };

        var wall = new WallJob
        {
            scale = Bootstrap.Param.wallScale * 0.5f,
            thresh = Bootstrap.Param.wallDistance,
            weight = Bootstrap.Param.wallWeight,
        };

        var separation = new SeparationJob 
        {
            separationWeight = Bootstrap.Param.separationWeight,
            neighborsFromEntity = GetBufferFromEntity<NeighborsEntityBuffer>(true),
            positionFromEntity = GetComponentDataFromEntity<Translation>(true),
        };

         var alignment = new AlignmentJob 
        {
            alignmentWeight = Bootstrap.Param.alignmentWeight,
            neighborsFromEntity = GetBufferFromEntity<NeighborsEntityBuffer>(true),
            velocityFromEntity = GetComponentDataFromEntity<Velocity>(true),
        };

        var cohesion = new CohesionJob 
        {
            cohesionWeight = Bootstrap.Param.cohesionWeight,
            neighborsFromEntity = GetBufferFromEntity<NeighborsEntityBuffer>(true),
            positionFromEntity = GetComponentDataFromEntity<Translation>(true),
        };

        var move = new MoveJob
        {
            dt = Time.DeltaTime,
            minSpeed = Bootstrap.Param.minSpeed,
            maxSpeed = Bootstrap.Param.maxSpeed,
        };

        inputDeps = neighbors.Schedule(this, inputDeps);
        inputDeps = wall.Schedule(this, inputDeps);
        inputDeps = separation.Schedule(this, inputDeps);
        inputDeps = alignment.Schedule(this, inputDeps);
        inputDeps = cohesion.Schedule(this, inputDeps);
        inputDeps = move.Schedule(this, inputDeps);
        return inputDeps;
    }
}