# Duckietown Safe Navigation — System Graph

This page contains a detailed, visual graph of the project suitable for quick inspection in VS Code (Mermaid).

Notes
- Replace <veh> with your vehicle name (VEHICLE_NAME).
- Optional modules are included; enable/disable them via `packages/duckietown_demos/launch/master.launch`.
- Topics shown reflect the effective remaps in the master launch.

## Mermaid diagram

```mermaid
flowchart TD
  %% Namespacing
  classDef opt fill:#f6f8fa,stroke:#ccd1d9,color:#666,font-size:12px
  classDef pkg fill:#eef7ff,stroke:#8fbdf1,color:#0b3d91,font-weight:bold
  classDef node fill:#fff,stroke:#aaa,color:#222
  classDef topic fill:#fff7e6,stroke:#e6b36b,color:#5a3d00,font-size:11px

  %% CAMERA
  subgraph pkg_camera[packages/camera]
    class pkg_camera pkg
    CAM["/<veh>/camera_node (camera_driver_node.py)"]:::node
    CAM_IMG["~image/compressed"]:::topic
    CAM_CI["~camera_info"]:::topic
    CAM -- publishes --> CAM_IMG
    CAM -- publishes --> CAM_CI
  end

  %% IMAGE PROCESSING (optional)
  subgraph pkg_image_proc[packages/image_processing]
    class pkg_image_proc pkg
    DEC["/<veh>/image_processing/decoder_node"]:::node
    DEC_IN["~image_in (compressed)"]:::topic
    DEC_OUT["~image_out (raw)"]:::topic
    DEC_IN -. optional .- CAM_IMG
    DEC -- publishes -. optional .-> DEC_OUT

    REC["/<veh>/image_processing/rectifier_node"]:::node
    REC_IN["~image_in (compressed)"]:::topic
    REC_CI_IN["~camera_info_in"]:::topic
    REC_IMG_OUT["~image_out (rectified)"]:::topic
    REC_CI_OUT["~camera_info_out"]:::topic
    REC_IN -. optional .- CAM_IMG
    REC_CI_IN -. optional .- CAM_CI
    REC -- publishes -. optional .-> REC_IMG_OUT
    REC -- publishes -. optional .-> REC_CI_OUT
  end

  %% ANTI INSTAGRAM (optional)
  subgraph pkg_ai[packages/anti_instagram]
    class pkg_ai pkg
    AI["/<veh>/anti_instagram_node"]:::node
    AI_THR["~thresholds"]:::topic
    AI -- publishes -. optional .-> AI_THR
  end

  %% LINE DETECTOR
  subgraph pkg_line_detector[packages/line_detector]
    class pkg_line_detector pkg
    LD["/<veh>/line_detector_node"]:::node
    LD_IN_IMG["~image/compressed"]:::topic
    LD_THR["~thresholds (from anti_instagram)"]:::topic
    LD_SEGS["~segment_list"]:::topic
    CAM_IMG --> LD_IN_IMG
    AI_THR -. optional .-> LD_THR
    LD -- publishes --> LD_SEGS
  end

  %% GROUND PROJECTION
  subgraph pkg_ground_projection[packages/ground_projection]
    class pkg_ground_projection pkg
    GP["/<veh>/ground_projection_node"]:::node
    GP_IN["~lineseglist_in"]:::topic
    GP_CI["~camera_info"]:::topic
    GP_OUT["~lineseglist_out"]:::topic
    LD_SEGS --> GP_IN
    CAM_CI --> GP_CI
    GP -- publishes --> GP_OUT
  end

  %% LANE FILTER
  subgraph pkg_lane_filter[packages/lane_filter]
    class pkg_lane_filter pkg
    LF["/<veh>/lane_filter_node"]:::node
    LF_IN["~segment_list"]:::topic
    LF_CMD["~car_cmd"]:::topic
    LF_POSE["~lane_pose"]:::topic
    LF_DEBUG["~seglist_filtered, ~belief_img"]:::topic
    GP_OUT --> LF_IN
    LF -- publishes --> LF_POSE
  end

  %% STOP LINE FILTER
  subgraph pkg_stop_line_filter[packages/stop_line_filter]
    class pkg_stop_line_filter pkg
    SLF["/<veh>/stop_line_filter_node"]:::node
    SLF_IN_SEGS["~segment_list"]:::topic
    SLF_IN_POSE["~lane_pose"]:::topic
    SLF_OUT["~stop_line_reading"]:::topic
    GP_OUT --> SLF_IN_SEGS
    LF_POSE --> SLF_IN_POSE
    SLF -- publishes --> SLF_OUT
  end

  %% VEHICLE DETECTION & FILTER
  subgraph pkg_vehicle_detection[packages/vehicle_detection]
    class pkg_vehicle_detection pkg
    VD["/<veh>/vehicle_detection_node"]:::node
    VD_IMG["~image (compressed)"]:::topic
    VD_CENTERS["~centers (VehicleCorners)"]:::topic
    VD_DET_IMG["~debug/detection_image"]:::topic

    VF["/<veh>/vehicle_filter_node"]:::node
    VF_CENTERS["~centers"]:::topic
    VF_CI["~camera_info"]:::topic
    VF_MODE["~mode (FSMState)"]:::topic
    VF_VSL["~virtual_stop_line (StopLineReading)"]:::topic

    CAM_IMG --> VD_IMG
    VD -- publishes --> VD_CENTERS
    VD_CENTERS --> VF_CENTERS
    CAM_CI --> VF_CI
  end

  %% APRILTAG DETECTION
  subgraph pkg_apriltag[packages/apriltag]
    class pkg_apriltag pkg
    ATD["/<veh>/apriltag_detector_node"]:::node
    ATD_IMG["~image (compressed)"]:::topic
    ATD_CI["~camera_info"]:::topic
    ATD_DET["~detections"]:::topic
    CAM_IMG --> ATD_IMG
    CAM_CI --> ATD_CI
    ATD -- publishes --> ATD_DET

    ATP["/<veh>/apriltag_postprocessing_node"]:::node
    ATP_IN["~detections"]:::topic
    ATP_OUT["~apriltags_out (AprilTagsWithInfos)"]:::topic
    ATP_PARK["~apriltags_parking (Bool)"]:::topic
    ATP_INT["~apriltags_intersection (Bool)"]:::topic
    ATD_DET --> ATP_IN
  end

  %% LANE CONTROL
  subgraph pkg_lane_control[packages/lane_control]
    class pkg_lane_control pkg
    LC["/<veh>/lane_controller_node"]:::node
    LC_IN_POSE["~lane_pose"]:::topic
    LC_IN_STOP["~stop_line_reading"]:::topic
    LC_IN_OBS["~obstacle_distance_reading (from VF)"]:::topic
    LC_IN_ATD["~apriltag_detections"]:::topic
    LC_WCE["~wheels_cmd (executed feedback)"]:::topic
    LC_CMD["~car_cmd (Twist2DStamped)"]:::topic

    LF_POSE --> LC_IN_POSE
    SLF_OUT --> LC_IN_STOP
    VF_VSL --> LC_IN_OBS
    ATD_DET --> LC_IN_ATD

    LC -- publishes --> LC_CMD
  end

  %% FSM & LED
  subgraph pkg_fsm_led[packages/fsm & led_*]
    class pkg_fsm_led pkg
    FSM["/<veh>/fsm_node"]:::node
    FSM_MODE["~mode (FSMState)"]:::topic
    FSM -- publishes --> FSM_MODE

    LED["/<veh>/led_emitter_node (svc set_pattern)"]:::node
    LPS["/<veh>/led_pattern_switch_node"]:::node
    LC -. call svc .-> LED

    FSM_MODE --> VF_MODE
    FSM_MODE --> LF
    FSM_MODE --> LC
  end

  %% OBJECT DETECTION (NN)
  subgraph pkg_object_detection[packages/object_detection]
    class pkg_object_detection pkg
    OBJ["/<veh>/object_detection_node"]:::node
    OBJ_IMG["/<veh>/object_detection_node/image/compressed (viz)"]:::topic
    OBJ_CMD["/<veh>/car_cmd_switch_node/cmd"]:::topic
    CAM_IMG --> OBJ
    OBJ -- publishes --> OBJ_CMD
  end

  %% CONTROL SWITCH + WHEELS (external)
  subgraph external_actuation[External Duckietown nodes]
    class external_actuation opt
    CARCMD["/<veh>/car_cmd_switch_node/cmd"]:::topic
    WD["/<veh>/wheels_driver_node"]:::node
    WD_CMD["/<veh>/wheels_cmd"]:::topic
    WD_EXEC["/<veh>/wheels_cmd_executed"]:::topic
    LC_CMD --> CARCMD
    OBJ_CMD --> CARCMD
    CARCMD --> WD_CMD
    WD_CMD --> WD
    WD -- publishes --> WD_EXEC
    WD_EXEC --> LC_WCE
  end

  %% VISUALIZATION TOOLS (optional)
  subgraph pkg_visualization[packages/visualization_tools]
    class pkg_visualization pkg
    LSV["/<veh>/line_segment_visualizer_node"]:::node
    LPV["/<veh>/lane_pose_visualizer_node"]:::node
    GP_OUT -. optional .-> LSV
    LF_POSE -. optional .-> LPV
  end
```

## How to view
- In VS Code: open this file and use the built-in Markdown preview; Mermaid renders inline.
- For Graphviz, see `graphs/system_graph.dot` and instructions inside that file.
