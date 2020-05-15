using Unity.Entities;
using Unity.Mathematics;
using Unity.Transforms;
using UnityEngine;

public class Bootstrap : MonoBehaviour 
{   
    public GameObject Prefab;

    public static Bootstrap Instance { get; private set; }
    public static Param Param { get { return Instance.param; } }

    [SerializeField] int boidCount = 100;
    [SerializeField] float3 boidScale = new float3(0.1f, 0.1f, 0.3f);
    [SerializeField] Param param;

    void Awake()
    {
        Instance = this;
    }

    void Start()
    {
        var settings = GameObjectConversionSettings.FromWorld(World.DefaultGameObjectInjectionWorld, null);
        var prefab = GameObjectConversionUtility.ConvertGameObjectHierarchy(Prefab, settings);
        var entityManager = World.DefaultGameObjectInjectionWorld.EntityManager;
        var random = new Unity.Mathematics.Random(853);

        for (int i = 0; i < boidCount; ++i)
        {
            var instance = entityManager.Instantiate(prefab);
            var position = transform.TransformPoint(random.NextFloat3(1f));

            entityManager.SetComponentData(instance, new Translation {Value = position});
            entityManager.SetComponentData(instance, new Rotation { Value = quaternion.identity });
            entityManager.SetComponentData(instance, new NonUniformScale { Value = new float3(boidScale.x, boidScale.y, boidScale.z) });
            entityManager.SetComponentData(instance, new Velocity { Value = random.NextFloat3Direction() * param.initSpeed });
            entityManager.SetComponentData(instance, new Acceleration { Value = float3.zero });
            
            // Dynamic Buffer の追加
            entityManager.AddBuffer<NeighborsEntityBuffer>(instance);
        }
    }

    void OnDrawGizmos()
    {
        if (!param) return;
        Gizmos.color = Color.green;
        Gizmos.DrawWireCube(Vector3.zero, Vector3.one * param.wallScale);
    }
}