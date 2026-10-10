import csv
import io
from datetime import datetime, timedelta


AUDIT_EVENT_TYPES = {
    'login', 'logout', 'create', 'delete', 'settings', 'update',
    'approve', 'deny', 'fulfill', 'add', 'remove', 'clear', 'reset',
    'factory_reset', 'export', 'archive', 'restore',
}

REPORT_TITLES = {
    'village': 'Village-Wide Security Summary Report',
    'incident': 'Security Incident Report',
    'frequency': 'Alert Frequency Analytics Report',
    'surveillance': 'Surveillance System Health Report',
}


def _severity(event_type):
    normalized = (event_type or '').lower()
    if any(keyword in normalized for keyword in ('critical', 'unknown', 'alert', 'error')):
        return 'High'
    if any(keyword in normalized for keyword in ('motion', 'object', 'detection')):
        return 'Warning'
    return 'Normal'


def _event_severity(event):
    value = getattr(event, 'severity', None)
    if value:
        normalized = str(value).lower()
        if normalized in {'critical', 'high'}:
            return 'High'
        if normalized in {'warning', 'medium'}:
            return 'Warning'
        if normalized in {'low', 'normal'}:
            return 'Normal'
    return _severity(getattr(event, 'event_type', None))


def _metadata_zone(event):
    event_metadata = getattr(event, 'event_metadata', None) or ''
    if not event_metadata:
        return None
    try:
        import json
        payload = json.loads(event_metadata)
        if isinstance(payload, dict):
            zone_value = payload.get('camera_zone') or payload.get('zone')
            if zone_value:
                return zone_value
    except (TypeError, ValueError):
        pass
    return None


def _resolved_zone(event):
    camera_zone = (getattr(event.camera, 'zone', None) if getattr(event, 'camera', None) else None) or ''
    metadata_zone = _metadata_zone(event) or ''
    camera_name = (getattr(event.camera, 'name', None) if getattr(event, 'camera', None) else None) or ''
    source_name = (getattr(event, 'source_name', None) or '')

    if camera_zone:
        return camera_zone
    if metadata_zone and metadata_zone.lower() not in {camera_name.lower(), source_name.lower()}:
        return metadata_zone
    return 'Main Area'


def _matches_zone(source_name, zone, event=None):
    if not zone or zone == 'all':
        return True

    target_zone = zone.strip().lower()
    candidate_source = (source_name or '').lower()
    metadata_zone = (_metadata_zone(event) or '').lower() if event else ''
    
    event_camera_zone = ''
    if event and getattr(event, 'camera', None):
        event_camera_zone = (getattr(event.camera, 'zone', '') or '').lower()

    combined_text = f"{candidate_source} {metadata_zone} {event_camera_zone}".strip()
    
    zone_aliases = {
        'gate1': ('gate 1', 'gate1', 'cam 1', 'entrance'),
        'north': ('north', 'cam 2', 'perimeter'),
        'clubhouse': ('clubhouse', 'amenities', 'cam 3'),
    }
    aliases = zone_aliases.get(target_zone, (target_zone,))
    return any(alias in combined_text for alias in aliases)


def _matches_severity(event_severity, severity):
    if not severity or severity == 'all':
        return True
    if severity == 'critical':
        return event_severity == 'High'
    if severity == 'normal':
        return event_severity == 'Normal'
    return True


def compute_patrol_priority_summary(events):
    """Score each zone/camera by recent activity to guide patrol prioritization."""
    zone_stats = {}

    for event in events or []:
        zone = (event.get('zone') or event.get('location') or 'Unassigned').strip() or 'Unassigned'
        camera = event.get('camera_name') or 'Unknown camera'
        key = (zone, camera)
        stats = zone_stats.setdefault(key, {
            'zone': zone,
            'camera_name': camera,
            'total_triggers': 0,
            'high_severity': 0,
            'unknown_faces': 0,
            'crowd_events': 0,
            'vehicle_crossings': 0,
            'risk_score': 0,
            'event_types': {},
        })

        stats['total_triggers'] += 1
        if event.get('severity') == 'High':
            stats['high_severity'] += 1

        event_type = str(event.get('event_type') or '').lower()
        if 'unknown' in event_type:
            stats['unknown_faces'] += 1
        if 'crowd' in event_type:
            stats['crowd_events'] += 1
        if 'vehicle' in event_type or 'crossing' in event_type:
            stats['vehicle_crossings'] += 1

        if 'critical' in event_type or 'alert' in event_type:
            stats['risk_score'] += 12
        elif 'unknown' in event_type:
            stats['risk_score'] += 9
        elif 'crowd' in event_type:
            stats['risk_score'] += 6
        elif 'vehicle' in event_type or 'crossing' in event_type:
            stats['risk_score'] += 5
        elif 'motion' in event_type:
            stats['risk_score'] += 2
        elif 'recogn' in event_type:
            stats['risk_score'] += 1
        else:
            stats['risk_score'] += 1

        stats['risk_score'] += stats['high_severity'] * 2
        stats['risk_score'] += stats['unknown_faces'] * 3
        stats['risk_score'] += stats['crowd_events'] * 2
        stats['risk_score'] += stats['vehicle_crossings'] * 2

        event_label = event.get('event_type') or 'Other'
        stats['event_types'][event_label] = stats['event_types'].get(event_label, 0) + 1

    summary = []
    for entry in zone_stats.values():
        risk_score = entry['risk_score']
        if risk_score >= 24:
            risk_level = 'High'
            patrol_recommendation = 'Escalate physical patrols and increase monitoring in this zone.'
        elif risk_score >= 10:
            risk_level = 'Medium'
            patrol_recommendation = 'Maintain active patrol checks and review the area during peak hours.'
        else:
            risk_level = 'Low'
            patrol_recommendation = 'Routine patrol cadence is adequate for this location.'

        strongest_event = max(entry['event_types'].items(), key=lambda item: item[1])[0] if entry['event_types'] else 'Routine activity'
        summary.append({
            'zone': entry['zone'],
            'camera_name': entry['camera_name'],
            'total_triggers': entry['total_triggers'],
            'high_severity': entry['high_severity'],
            'risk_score': risk_score,
            'risk_level': risk_level,
            'anomaly_flag': 'Yes' if risk_level != 'Low' else 'No',
            'patrol_recommendation': patrol_recommendation,
            'top_event': strongest_event,
        })

    summary.sort(key=lambda item: item['risk_score'], reverse=True)
    return summary


def build_report(
    event_log_model,
    report_type='incident',
    timeframe='7d',
    zone='all',
    severity='all',
    start_date=None,
    end_date=None,
    now=None,
    camera_model=None,
):
    """Build the admin report context from security events in EventLog."""
    now = now or datetime.now()
    report_type = report_type if report_type in REPORT_TITLES else 'incident'
    timeframe = timeframe if timeframe in {'24h', '7d', '30d', 'custom'} else '7d'
    severity = severity if severity in {'all', 'critical', 'normal'} else 'all'

    # Extract dynamic camera zones from database
    available_zones = []
    if camera_model:
        try:
            cameras = camera_model.query.all()
            for cam in cameras:
                cam_zone = (getattr(cam, 'zone', None) or '').strip()
                if cam_zone and cam_zone not in available_zones:
                    available_zones.append(cam_zone)
        except Exception:
            pass

    if not available_zones:
        available_zones = ['Main Entrance', 'North Perimeter', 'Clubhouse & Amenities']

    since = None
    until = None

    if timeframe == 'custom' and start_date and end_date:
        try:
            since = datetime.strptime(start_date, '%Y-%m-%d')
            until = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
        except (ValueError, TypeError):
            timeframe = '7d'

    if not since:
        timeframe_days = {'24h': 1, '7d': 7, '30d': 30}.get(timeframe, 7)
        since = now - timedelta(days=timeframe_days)

    filters = [
        event_log_model.timestamp >= since,
        ~event_log_model.event_type.in_(AUDIT_EVENT_TYPES),
    ]
    if until:
        filters.append(event_log_model.timestamp <= until)

    query = event_log_model.query.filter(*filters)
    events = query.order_by(event_log_model.timestamp.desc()).all()

    report_events = []
    for event in events:
        event_severity = _event_severity(event)
        if not _matches_zone(event.source_name, zone, event) or not _matches_severity(event_severity, severity):
            continue
        event_label = event.event_type or 'Unspecified event'
        if event_label.upper() == 'CRITICAL ALERT' and event.description:
            event_label = f'{event_label}: {event.description.removeprefix("Detected: ").strip()}'
        
        event_zone = _resolved_zone(event)
        cam_name = event.source_name or 'Unknown camera'

        report_events.append({
            'timestamp': event.timestamp.strftime('%Y-%m-%d %I:%M:%S %p') if event.timestamp else '-',
            'raw_timestamp': event.timestamp,
            'location': event_zone,
            'zone': event_zone,
            'camera_name': cam_name,
            'event_type': event_label,
            'confidence': f'{event.confidence * 100:.1f}%' if event.confidence is not None else '-',
            'severity': event_severity,
        })

    # Calculate KPI Summary Metrics
    recognized_count = sum(
        1 for event in report_events if 'recogn' in event['event_type'].lower()
    )
    unrecognized_count = sum(
        1 for event in report_events
        if any(keyword in event['event_type'].lower() for keyword in ('unknown', 'critical', 'alert'))
    )

    # Filtered Incident Logs for Incident Report
    if report_type == 'incident':
        filtered_events = [
            e for e in report_events
            if e['severity'] in {'High', 'Warning'} or 'unknown' in e['event_type'].lower()
        ]
    else:
        filtered_events = report_events

    # Frequency Analytics Breakdown by Zone / Camera Name
    patrol_priority_summary = []
    frequency_summary = []
    if report_type == 'frequency':
        location_data = {}
        for event in report_events:
            camera_label = event.get('camera_name') or event.get('location') or 'Unknown camera'
            key = (event['zone'], camera_label)
            if key not in location_data:
                location_data[key] = {
                    'zone': event['zone'],
                    'camera_name': camera_label,
                    'total_triggers': 0,
                    'high_severity': 0,
                    'hours': {},
                    'types': {},
                }
            location_data[key]['total_triggers'] += 1
            if event['severity'] == 'High':
                location_data[key]['high_severity'] += 1
            
            ts = event.get('raw_timestamp')
            if ts:
                hour_str = ts.strftime('%I:00 %p')
                location_data[key]['hours'][hour_str] = location_data[key]['hours'].get(hour_str, 0) + 1
            
            ev_type = event['event_type']
            location_data[key]['types'][ev_type] = location_data[key]['types'].get(ev_type, 0) + 1

        for (z, c), data in location_data.items():
            peak_h = max(data['hours'], key=data['hours'].get) if data['hours'] else 'N/A'
            top_t = max(data['types'], key=data['types'].get) if data['types'] else 'N/A'
            frequency_summary.append({
                'zone': z,
                'camera_name': c,
                'location': f"{z} ({c})",
                'total_triggers': data['total_triggers'],
                'high_severity': data['high_severity'],
                'peak_hour': peak_h,
                'top_event': top_t,
            })
        frequency_summary.sort(key=lambda x: x['total_triggers'], reverse=True)

        patrol_priority_summary = compute_patrol_priority_summary(report_events)
        priority_lookup = {(item['zone'], item['camera_name']): item for item in patrol_priority_summary}
        for row in frequency_summary:
            priority = priority_lookup.get((row['zone'], row['camera_name']), {})
            row['risk_score'] = priority.get('risk_score', 0)
            row['risk_level'] = priority.get('risk_level', 'Low')
            row['patrol_recommendation'] = priority.get('patrol_recommendation', 'Routine patrol cadence is adequate for this location.')
            row['anomaly_flag'] = priority.get('anomaly_flag', 'No')

        zone_totals = {}
        for item in patrol_priority_summary:
            zone_name = item['zone']
            zone_totals[zone_name] = zone_totals.get(zone_name, 0) + item['risk_score']

        zone_risk_chart = [
            {'label': zone, 'value': score}
            for zone, score in sorted(zone_totals.items(), key=lambda x: x[1], reverse=True)[:6]
        ]

        hourly_activity = {}
        for event in report_events:
            ts = event.get('raw_timestamp')
            if ts is None:
                continue
            hour_key = ts.strftime('%H:00')
            hourly_activity[hour_key] = hourly_activity.get(hour_key, 0) + 1
        hourly_activity_chart = [
            {'label': hour, 'value': count}
            for hour, count in sorted(hourly_activity.items())
        ]

        severity_counts = {'High': 0, 'Warning': 0, 'Normal': 0}
        for event in report_events:
            level = event.get('severity') or 'Normal'
            if level in severity_counts:
                severity_counts[level] += 1
        severity_chart = [
            {'label': label, 'value': value}
            for label, value in severity_counts.items()
            if value > 0
        ]
    else:
        zone_risk_chart = []
        hourly_activity_chart = []
        severity_chart = []

    # Surveillance System Health Breakdown
    surveillance_summary = []
    uptime_rate = '100.0%'
    offline_event_count = 0

    if camera_model:
        cameras = camera_model.query.all()
        for cam in cameras:
            cam_zone = getattr(cam, 'zone', 'Main Area') or 'Main Area'
            cam_events = [
                e for e in report_events
                if (e.get('camera_name') or '').lower() == cam.name.lower()
                or cam_zone.lower() in (e.get('zone') or '').lower()
            ]
            disconnects = sum(1 for e in cam_events if 'offline' in e['event_type'].lower() or 'disconnect' in e['event_type'].lower())
            offline_event_count += disconnects
            last_ts = cam_events[0]['timestamp'] if cam_events else 'No recent logs'
            status = 'Operational' if disconnects == 0 else f'{disconnects} Disconnect Alert(s)'
            surveillance_summary.append({
                'zone': cam_zone,
                'camera_name': cam.name,
                'status': status,
                'disconnect_events': disconnects,
                'last_activity': last_ts,
            })
        if cameras and offline_event_count > 0:
            total_evals = max(len(report_events), 1)
            calculated = max(0.0, 100.0 - (offline_event_count / total_evals * 100))
            uptime_rate = f'{calculated:.1f}%'
    else:
        uptime_rate = '99.8%' if len(report_events) > 0 else '100.0%'

    return {
        'selected_type': report_type,
        'timeframe': timeframe,
        'zone': zone,
        'severity': severity,
        'start_date': start_date or '',
        'end_date': end_date or '',
        'available_zones': available_zones,
        'report_title': REPORT_TITLES.get(report_type, REPORT_TITLES['village']),
        'total_triggers': len(report_events),
        'verified_residents': recognized_count,
        'unrecognized_alerts': unrecognized_count,
        'uptime_rate': uptime_rate,
        'incident_logs': filtered_events,
        'frequency_summary': frequency_summary,
        'patrol_priority_summary': patrol_priority_summary,
        'zone_risk_chart': zone_risk_chart,
        'hourly_activity_chart': hourly_activity_chart,
        'severity_chart': severity_chart,
        'surveillance_summary': surveillance_summary,
        'date_generated': now.strftime('%Y-%m-%d'),
        'doc_id': f'EBV-{now.strftime("%Y%m%d")}-{len(report_events):04d}',
    }


def generate_report_csv(report_data):
    """Generate a clean, standardized CSV representation of the report data."""
    output = io.StringIO()
    writer = csv.writer(output)

    report_type = report_data.get('selected_type', 'village')

    if report_type == 'frequency':
        writer.writerow(['Assigned Zone', 'Camera Name', 'Total Triggers', 'High Severity Alerts', 'Peak Hour', 'Primary Trigger Type', 'Risk Score', 'Patrol Priority', 'Patrol Recommendation'])
        for row in report_data.get('frequency_summary', []):
            writer.writerow([
                row.get('zone', '-'),
                row.get('camera_name', '-'),
                row.get('total_triggers', 0),
                row.get('high_severity', 0),
                row.get('peak_hour', '-'),
                row.get('top_event', '-'),
                row.get('risk_score', 0),
                row.get('risk_level', 'Low'),
                row.get('patrol_recommendation', '-'),
            ])
    elif report_type == 'surveillance':
        writer.writerow(['Assigned Zone', 'Camera Name', 'Operational Status', 'Disconnect Events', 'Last Recorded Activity'])
        for row in report_data.get('surveillance_summary', []):
            writer.writerow([
                row.get('zone', '-'),
                row.get('camera_name', '-'),
                row.get('status', '-'),
                row.get('disconnect_events', 0),
                row.get('last_activity', '-'),
            ])
    else:
        writer.writerow(['Timestamp', 'Assigned Zone', 'Camera Name', 'Event / Trigger Type', 'Confidence', 'Severity Status'])
        for row in report_data.get('incident_logs', []):
            writer.writerow([
                row.get('timestamp', '-'),
                row.get('location', '-'),
                row.get('camera_name', row.get('location', '-')),
                row.get('event_type', '-'),
                row.get('confidence', '-'),
                row.get('severity', '-'),
            ])

    return output.getvalue()