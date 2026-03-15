using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;

public class AgentManager : MonoBehaviour
{
    private ConsciousnessChannel _consciousnessChannel;
    private EmotionChannel _emotionChannel;

    public void Awake()
    {
        // Register Side Channels
        _consciousnessChannel = new ConsciousnessChannel();
        _emotionChannel = new EmotionChannel();
        
        SideChannelManager.RegisterSideChannel(_consciousnessChannel);
        SideChannelManager.RegisterSideChannel(_emotionChannel);
        
        Debug.Log("Side Channels Registered.");
    }

    public void OnDestroy()
    {
        // Unregister to prevent memory leaks
        if (_consciousnessChannel != null)
        {
            SideChannelManager.UnregisterSideChannel(_consciousnessChannel);
        }

        if (_emotionChannel != null)
        {
            SideChannelManager.UnregisterSideChannel(_emotionChannel);
        }
    }
}
