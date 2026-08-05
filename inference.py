def run_overtaking(video_path):

    print("Processing video:", video_path)

    import torch
    import torch.nn as nn
    import numpy as np
    import cv2
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
    import os

    vehicle_map = {
        2: "CAR",
        3: "BIKE",
        5: "BUS",
        7: "TRUCK"
    }

# ===========================
# LSTM MODEL
# ===========================

    class AdvancedOvertakingLSTM(nn.Module):

        def __init__(self, input_size, hidden_size, num_classes):
            super().__init__()

            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
                bidirectional=True
            )

            self.dropout = nn.Dropout(0.4)
            self.fc1 = nn.Linear(hidden_size * 2, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out


    # ===========================
    # LOAD MODEL
    # ===========================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdvancedOvertakingLSTM(6, 192, 2).to(device)

    model.load_state_dict(
        torch.load("model/overtaking_model.pth", map_location=device)
    )

    model.eval()


    # ===========================
    # YOLO + DEEPSORT
    # ===========================

    yolo_model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=30)


    # ===========================
    # VIDEO
    # ===========================


    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    processed_folder = "processed"
    os.makedirs(processed_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}_analysis.mp4"
    output_path = os.path.join(processed_folder, output_filename)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )


    # ===========================
    # MEMORY
    # ===========================

    track_memory = {}
    time_steps = 60


    # ===========================
    # EVENT VARIABLES
    # ===========================

    overtaking_active = False
    positive_count = 0
    negative_count = 0

    threshold = 0.6

    max_overtake_frames = int(fps * 5)
    overtake_frame_count = 0

    cooldown_frames  = int(fps * 2)
    cooldown_counter = 0

    max_dx = 0
    max_area_ratio = 0

    event_start_frame = 0

    behavior_label = ""
    result_display_frames = int(fps * 3)
    result_counter = 0

    frame_id = 0


    # ===========================
    # GLOBAL EVENT COOLDOWN
    # ===========================

    global_event_cooldown = 0
    min_frames_between_events = int(fps * 0.5)


    # ===========================
    # DRIVER SCORE
    # ===========================

    driver_score = 100
    safe_overtakes = 0
    rash_overtakes = 0


    # ===========================
    # NEW: PROBABILITY SMOOTHING
    # ===========================

    prob_history = []


    print("Starting inference...")


    # ===========================
    # MAIN LOOP
    # ===========================

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if global_event_cooldown > 0:
            global_event_cooldown -= 1


        results = yolo_model(frame)
        detections = []

        for r in results:
            for box in r.boxes:

                cls  = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in [2, 3, 5, 7]:

                    x1, y1, x2, y2 = box.xyxy[0]
                    w = x2 - x1
                    h = y2 - y1

                    detections.append(
                        ([float(x1), float(y1), float(w), float(h)], conf, cls)
                    )


        tracks = tracker.update_tracks(detections, frame=frame)
        active_track_ids = set()


        for track in tracks:

            if not track.is_confirmed():
                continue

            track_id = track.track_id
            active_track_ids.add(track_id)

            ltrb = track.to_ltrb()
            l, t = ltrb[0], ltrb[1]
            w    = ltrb[2] - ltrb[0]
            h    = ltrb[3] - ltrb[1]

            x_center = l + w / 2
            x_norm   = x_center / width
            area     = w * h

            if x_norm < 0.05 or x_norm > 0.95:
                continue

            if track_id not in track_memory:
                track_memory[track_id] = {
                    "first_x": x_norm,
                    "history": [],
                    "vehicle_type": vehicle_map.get(cls, "VEHICLE")
                }

            first_x = track_memory[track_id]["first_x"]
            history = track_memory[track_id]["history"]
            vehicle_type = track_memory[track_id]["vehicle_type"]

            dx = 0
            area_ratio = 0

            if len(history) > 0:
                prev_x, prev_area = history[-1]
                dx = x_norm - prev_x
                area_ratio = (area - prev_area) / (prev_area + 1e-6)

            history.append((x_norm, area))

            shift = x_norm - first_x
            passing_vehicle = abs(shift) > 0.25


            # ===========================
            # LSTM INFERENCE
            # ===========================

            if len(history) >= time_steps:

                seq_input = []

                for i in range(-time_steps, 0):

                    x_val, area_val = history[i]

                    if i > -time_steps:
                        px, pa = history[i - 1]
                        step_dx = x_val - px
                        step_area_ratio = (area_val - pa) / (pa + 1e-6)
                    else:
                        step_dx = 0
                        step_area_ratio = 0

                    rel_shift = x_val - first_x

                    seq_input.append([
                        x_val,
                        area_val,
                        step_dx,
                        step_area_ratio,
                        0,
                        rel_shift
                    ])

                seq_tensor = torch.tensor(
                    np.array(seq_input).reshape(1, time_steps, 6),
                    dtype=torch.float32
                ).to(device)

                with torch.no_grad():
                    output = model(seq_tensor)
                    probs = torch.softmax(output, dim=1)

                    # 🔥 SMOOTHING
                    prob_history.append(probs[0][1].item())
                    if len(prob_history) > 10:
                        prob_history.pop(0)

                    overtaking_prob = sum(prob_history) / len(prob_history)

                pred = 1 if overtaking_prob >= threshold else 0


                # ===========================
                # EVENT SMOOTHING
                # ===========================

                if pred == 1:
                    positive_count += 1
                    negative_count = 0
                    cooldown_counter = cooldown_frames
                else:
                    negative_count += 1
                    positive_count = 0


                # ===========================
                # START EVENT
                # ===========================

                if (
                    not overtaking_active
                    and positive_count >= 5
                    and overtaking_prob > 0.65
                    and global_event_cooldown <= 0
                ):
                    overtaking_active = True
                    overtake_frame_count = 0
                    max_dx = 0
                    max_area_ratio = 0
                    event_start_frame = frame_id

                    print("Overtaking STARTED")


                if overtaking_active:
                    overtake_frame_count += 1
                    max_dx = max(max_dx, abs(dx))
                    max_area_ratio = max(max_area_ratio, abs(area_ratio))


                # ===========================
                # END EVENT
                # ===========================

                if overtaking_active:

                    if pred == 0:
                        cooldown_counter -= 1

                    if cooldown_counter <= 0 or overtake_frame_count > max_overtake_frames:

                        duration = (frame_id - event_start_frame) / fps

                        if (
                            max_dx > 0.035
                            or max_area_ratio > 0.25
                            or duration < 1.5
                        ):
                            behavior_label = f"RASH OVERTAKE ({vehicle_type})"
                            rash_overtakes += 1
                            driver_score -= 20
                        else:
                            behavior_label = f"SAFE OVERTAKE ({vehicle_type})"
                            safe_overtakes += 1
                            driver_score += 5

                        driver_score = max(0, min(100, driver_score))

                        result_counter = result_display_frames
                        overtaking_active = False

                        global_event_cooldown = min_frames_between_events

                        print("Overtaking ENDED:", behavior_label)


                confidence_text = f"{overtaking_prob*100:.1f}%"


                # ===========================
                # LABEL LOGIC
                # ===========================

                if overtaking_active:
                    label = f"{vehicle_type} OVERTAKING ({confidence_text})"
                    color = (0,255,0)

                elif result_counter > 0:
                    label = behavior_label
                    color = (0,0,255) if "RASH" in behavior_label else (0,255,0)
                    result_counter -= 1

                elif passing_vehicle:
                    label = "PASSING VEHICLE"
                    color = (255,255,0)

                else:
                    label = f"NORMAL ({confidence_text})"
                    color = (0,0,255)


                cv2.rectangle(frame,(int(l),int(t)),(int(l+w),int(t+h)),color,2)

                cv2.putText(frame,label,(int(l),int(t)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)


        cv2.putText(frame,f"Driver Score: {driver_score}/100",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        cv2.putText(frame,f"Safe: {safe_overtakes} Rash: {rash_overtakes}",
                    (30,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


        # out.write(frame)
        # cv2.imshow("Overtaking Detection", frame)

        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break


    cap.release()
    out.release()
    # cv2.destroyAllWindows()

    print("Final Score:", driver_score)
    if driver_score >= 85:
        driving_style = "SAFE"
    elif driver_score >= 60:
        driving_style = "MODERATE"
    else:
        driving_style = "AGGRESSIVE"

    print("\n===== DRIVER REPORT =====")
    print(f"Final Score     : {driver_score}/100")
    print(f"Safe Overtakes  : {safe_overtakes}")
    print(f"Rash Overtakes  : {rash_overtakes}")
    print(f"Driving Style   : {driving_style}")
    print("==========================")
    print("Safe:", safe_overtakes, "Rash:", rash_overtakes)

    with open("driver_report.txt", "w") as f:
        f.write("===== DRIVER REPORT =====\n")
        f.write(f"Driver Score     : {driver_score}/100\n")
        f.write(f"Safe Overtakes   : {safe_overtakes}\n")
        f.write(f"Rash Overtakes   : {rash_overtakes}\n")
        f.write(f"Driving Style    : {driving_style}\n")
        f.write("=========================\n")

    return {
        "score": driver_score,
        "safe": safe_overtakes,
        "rash": rash_overtakes,
        "video_filename": output_filename
    }
