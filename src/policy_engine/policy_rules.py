import json
import os
import re

POLICY_FILE = 'models/policy_config.json'


def get_rules():
    """
    Define security policy rules with adaptive thresholds and XAI-driven
    feature-to-action mappings per Paper §5.1.5 and §7.5.
    """
    # Correction #12: Add try/except validation for JSON loading
    if os.path.exists(POLICY_FILE):
        try:
            with open(POLICY_FILE, 'r') as f:
                rules = json.load(f)
                if isinstance(rules, dict) and 'high_risk' in rules:
                    return rules
        except (json.JSONDecodeError, IOError):
            pass  # Fall through to default rules

    rules = {
        # Risk-threshold tiers (Paper §5.1.5)
        'critical_risk': {'threshold': 0.95, 'action': 'isolate_subnet'},
        'high_risk':     {'threshold': 0.80, 'action': 'block_connection'},
        'medium_risk':   {'threshold': 0.50, 'action': 'require_mfa'},
        'low_risk':      {'threshold': 0.30, 'action': 'rate_limit'},
        'minimal_risk':  {'threshold': 0.00, 'action': 'allow'},
    }
    return rules


def _normalize_feature_name(name):
    """
    Correction #6 & #13: Normalize feature names for robust matching.
    Converts to lowercase, replaces spaces/hyphens with underscores,
    strips extra whitespace.
    """
    name = str(name).strip().lower()
    name = re.sub(r'[\s\-]+', '_', name)
    name = re.sub(r'_+', '_', name)
    return name


# XAI feature → targeted action mapping (Paper §5.1.5, §7.5)
# Keys are normalized for robust matching (Correction #6)
FEATURE_ACTION_MAP = {
    'dns_query':           'rate_limit_dns',
    'flow_duration':       'bandwidth_throttle',
    'syn_flag_count':      'block_syn_flood',
    'fwd_packet_length':   'bandwidth_throttle',
    'dst_port':            'restrict_port',
    'src_bytes':           'bandwidth_throttle',
    'payload_length':      'deep_packet_inspect',
    'protocol_type':       'restrict_protocol',
    'dst_host_count':      'restrict_lateral_movement',
    'srv_count':           'rate_limit_service',
    'bwd_packet_length':   'bandwidth_throttle',
    'total_fwd_packets':   'rate_limit_service',
    'destination_port':    'restrict_port',
    'source_port':         'restrict_port',
}


def get_feature_action_map():
    """Return the XAI feature → targeted action mapping."""
    return FEATURE_ACTION_MAP


def update_rules(new_rules):
    """Save updated rules to disk."""
    # Correction #11: Ensure directory exists before saving
    os.makedirs(os.path.dirname(POLICY_FILE) if os.path.dirname(POLICY_FILE) else 'models', exist_ok=True)
    with open(POLICY_FILE, 'w') as f:
        json.dump(new_rules, f, indent=4)