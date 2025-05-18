import pandas as pd
import numpy as np
from corner_detection import detect_corners
from math import radians, sin, cos, sqrt, atan2
import re

def find_nearest_point(lat, lon, df):
    # Find the index of the point in df closest to (lat, lon)
    dists = np.sqrt((df['latitude'].astype(float) - lat)**2 + (df['longitude'].astype(float) - lon)**2)
    return dists.idxmin()

def get_distance_from_start(df, idx):
    if 'distance' in df.columns:
        return float(df.iloc[idx]['distance'])
    return None

def get_lap_time(df):
    if 'timestamp' in df.columns:
        t0 = float(df['timestamp'].iloc[0])
        t1 = float(df['timestamp'].iloc[-1])
        return t1 - t0
    if 'time' in df.columns:
        t0 = float(df['time'].iloc[0])
        t1 = float(df['time'].iloc[-1])
        return t1 - t0
    return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def feedback_template(msg):
    # Remove numbers (including decimals and minus signs) for template comparison
    return re.sub(r'[-+]?[0-9]*\.?[0-9]+', 'X', msg)

def filter_spatially_close(feedbacks, min_dist=100):
    # Only keep the most significant feedback (largest deviation) for each type/message template in a 100m radius
    filtered = []
    for fb in feedbacks:
        template = feedback_template(fb['feedback'][0]) if fb['feedback'] else ''
        key = (fb['type'], template)
        too_close = False
        for i, sel in enumerate(filtered):
            sel_template = feedback_template(sel['feedback'][0]) if sel['feedback'] else ''
            sel_key = (sel['type'], sel_template)
            if key == sel_key and haversine(fb['latitude'], fb['longitude'], sel['latitude'], sel['longitude']) < min_dist:
                # Keep the one with the largest deviation if available
                if 'total_dev' in fb and 'total_dev' in sel:
                    if fb['total_dev'] > sel['total_dev']:
                        filtered[i] = fb
                too_close = True
                break
        if not too_close:
            filtered.append(fb)
    # Remove the 'total_dev' key if present
    for fb in filtered:
        if 'total_dev' in fb:
            del fb['total_dev']
    return filtered

def calculate_heading(lat1, lon1, lat2, lon2):
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    x = sin(dLon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    heading = atan2(x, y)
    heading = np.degrees(heading)
    return (heading + 360) % 360

def detect_significant_corners(df, heading_threshold=20, num_corners=14):
    # Like detect_corners, but only keep corners with heading change > threshold
    lats = df['latitude'].astype(float).values
    lons = df['longitude'].astype(float).values
    headings = []
    for i in range(1, len(lats)):
        headings.append(calculate_heading(lats[i-1], lons[i-1], lats[i], lons[i]))
    headings = np.array([0] + headings)
    delta_heading = np.abs(np.diff(headings, n=5, prepend=[headings[0]]*5))
    delta_heading = np.where(delta_heading > 180, 360 - delta_heading, delta_heading)
    # Only keep corners above threshold
    indices = np.where(delta_heading > heading_threshold)[0]
    # If too many, keep the top num_corners
    if len(indices) > num_corners:
        top_indices = np.argpartition(-delta_heading[indices], num_corners)[:num_corners]
        indices = indices[top_indices]
    indices = np.sort(indices)
    return indices.tolist()

def generate_corner_feedback(optimal, real):
    # Use adaptive corner detection
    corner_indices = detect_significant_corners(optimal)
    feedback_rows = []
    for i, corner_idx in enumerate(corner_indices):
        opt_row = optimal.iloc[corner_idx]
        lat, lon = float(opt_row['latitude']), float(opt_row['longitude'])
        real_idx = find_nearest_point(lat, lon, real)
        real_row = real.iloc[real_idx]
        feedbacks = []
        # Braking Point Feedback
        opt_dist = get_distance_from_start(optimal, corner_idx)
        real_dist = get_distance_from_start(real, real_idx)
        if opt_dist is not None and real_dist is not None:
            brake_diff = real_dist - opt_dist
            if abs(brake_diff) > 5:
                if brake_diff > 0:
                    feedbacks.append(f"You braked {brake_diff:.1f} meters later than optimal. Try to brake at the reference marker for better entry speed.")
                else:
                    feedbacks.append(f"You braked {abs(brake_diff):.1f} meters earlier than optimal. Try to delay braking for better entry speed.")
        # Entry Speed Feedback
        opt_speed = float(opt_row['speed'])
        real_speed = float(real_row['speed'])
        speed_diff = real_speed - opt_speed
        if abs(speed_diff) > 2:
            if speed_diff > 0:
                feedbacks.append(f"Your entry speed was {speed_diff:.1f} km/h higher than optimal. This may affect your line and exit speed.")
            else:
                feedbacks.append(f"Your entry speed was {abs(speed_diff):.1f} km/h lower than optimal. You can carry more speed into the corner.")
        # Apex Miss Feedback (if possible)
        apex_miss = np.sqrt((lat - float(real_row['latitude']))**2 + (lon - float(real_row['longitude']))**2) * 111000
        if apex_miss > 2:
            feedbacks.append(f"You missed the apex by {apex_miss:.1f} meters. Focus on hitting the apex for better cornering.")
        # Throttle Application Feedback (using acceleration x if available)
        if 'acceleration x' in real.columns and 'acceleration x' in optimal.columns:
            opt_accel = float(opt_row['acceleration x'])
            real_accel = float(real_row['acceleration x'])
            accel_diff = real_accel - opt_accel
            if abs(accel_diff) > 0.5:
                if accel_diff > 0:
                    feedbacks.append(f"You applied throttle {accel_diff:.2f} m/s² more aggressively than optimal. Early throttle can cause understeer on exit.")
                else:
                    feedbacks.append(f"You applied throttle {abs(accel_diff):.2f} m/s² less than optimal. Try to accelerate earlier for better exit speed.")
        # Steering Smoothness (using gyro y if available)
        if 'gyroy' in real.columns and 'gyroy' in optimal.columns:
            opt_gyro = float(opt_row['gyroy'])
            real_gyro = float(real_row['gyroy'])
            gyro_diff = abs(real_gyro) - abs(opt_gyro)
            if abs(gyro_diff) > 0.2:
                feedbacks.append(f"Your steering input was {gyro_diff:.2f} more aggressive than optimal. Smoother steering will help maintain tire grip.")
        # G-Force Management
        if 'g force y' in real.columns and 'g force y' in optimal.columns:
            opt_g = float(opt_row['g force y'])
            real_g = float(real_row['g force y'])
            g_diff = real_g - opt_g
            if abs(g_diff) > 0.1:
                feedbacks.append(f"Your lateral G-force exceeded optimal by {g_diff:.2f}g. This may indicate overdriving or loss of grip.")
        # Exit Speed Feedback (look ahead a few points if possible)
        lookahead = 5
        if corner_idx + lookahead < len(optimal) and real_idx + lookahead < len(real):
            opt_exit_speed = float(optimal.iloc[corner_idx + lookahead]['speed'])
            real_exit_speed = float(real.iloc[real_idx + lookahead]['speed'])
            exit_diff = real_exit_speed - opt_exit_speed
            if abs(exit_diff) > 2:
                if exit_diff < 0:
                    feedbacks.append(f"Your exit speed was {abs(exit_diff):.1f} km/h lower than optimal. Focus on earlier throttle and better line through the corner.")
                else:
                    feedbacks.append(f"Your exit speed was {exit_diff:.1f} km/h higher than optimal. Good job carrying speed out of the corner!")
        # Only add feedback if there is something meaningful
        if feedbacks:
            feedback_rows.append({
                'type': 'corner',
                'corner': i+1,
                'latitude': lat,
                'longitude': lon,
                'feedback': feedbacks
            })
    return feedback_rows

def generate_best_worst_feedback(optimal, real, n_best=5, n_weak=5):
    deviations = []
    for idx, opt_row in optimal.iterrows():
        lat, lon = float(opt_row['latitude']), float(opt_row['longitude'])
        real_idx = find_nearest_point(lat, lon, real)
        real_row = real.iloc[real_idx]
        opt_speed = float(opt_row['speed'])
        real_speed = float(real_row['speed'])
        speed_diff = abs(real_speed - opt_speed)
        opt_g = float(opt_row['g force y']) if 'g force y' in opt_row and 'g force y' in real_row else 0
        real_g = float(real_row['g force y']) if 'g force y' in opt_row and 'g force y' in real_row else 0
        g_diff = abs(real_g - opt_g)
        total_dev = speed_diff + g_diff
        deviations.append({
            'idx': idx,
            'lat': lat,
            'lon': lon,
            'speed_diff': speed_diff,
            'g_diff': g_diff,
            'total_dev': total_dev,
            'real_idx': real_idx
        })
    sorted_devs = sorted(deviations, key=lambda x: x['total_dev'])
    best_points = sorted_devs[:n_best]
    weak_points = sorted_devs[-n_weak:]
    feedback_rows = []
    for pt in best_points:
        feedback_rows.append({
            'type': 'positive',
            'latitude': pt['lat'],
            'longitude': pt['lon'],
            'feedback': [
                f"Excellent job! You matched the optimal line and speed here (Δspeed: {pt['speed_diff']:.1f} km/h, ΔG: {pt['g_diff']:.2f}). Keep it up!"
            ],
            'total_dev': pt['total_dev']
        })
    for pt in weak_points:
        feedbacks = []
        if pt['speed_diff'] > 2:
            feedbacks.append(f"You lost {pt['speed_diff']:.1f} km/h compared to optimal. Focus on smoother throttle/braking here.")
        if pt['g_diff'] > 0.1:
            feedbacks.append(f"Lateral G-force deviation is {pt['g_diff']:.2f}. Try to maintain better grip and line.")
        if not feedbacks:
            feedbacks.append("This section can be improved. Review your line and inputs.")
        feedback_rows.append({
            'type': 'constructive',
            'latitude': pt['lat'],
            'longitude': pt['lon'],
            'feedback': feedbacks,
            'total_dev': pt['total_dev']
        })
    # Filter spatially close feedbacks (for best and worst points)
    feedback_rows = filter_spatially_close(feedback_rows, min_dist=100)
    return feedback_rows

def generate_feedback(optimal, real, n_best=5, n_weak=5):
    # Lap time based scoring
    opt_time = get_lap_time(optimal)
    real_time = get_lap_time(real)
    score = 100
    time_penalty = 0
    if opt_time and real_time:
        time_ratio = real_time / opt_time
        if time_ratio > 1:
            time_penalty = int((time_ratio - 1) * 100)  # 1% slower = 1pt off
            score -= time_penalty
    # Corner feedbacks
    corner_feedbacks = generate_corner_feedback(optimal, real)
    # Best/worst feedbacks
    best_worst_feedbacks = generate_best_worst_feedback(optimal, real, n_best, n_weak)
    # Score penalties for worst points
    for fb in best_worst_feedbacks:
        if fb['type'] == 'constructive':
            score -= 2  # small penalty for each weak point
        if fb['type'] == 'positive':
            score += 1  # small bonus for each best point
    score = max(0, min(100, score))
    if not corner_feedbacks:
        summary = f"Drive rating: {score}/100. Lap time: {real_time:.2f}s (optimal: {opt_time:.2f}s). No significant corners detected on this track. Feedback is based on overall driving line and performance."
    else:
        summary = f"Drive rating: {score}/100. Lap time: {real_time:.2f}s (optimal: {opt_time:.2f}s). {n_best} sections were very close to optimal, {n_weak} sections need improvement."
    return {
        'score': score,
        'summary': summary,
        'lap_time': real_time,
        'optimal_time': opt_time,
        'feedback': corner_feedbacks + best_worst_feedbacks
    } 