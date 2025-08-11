import hashlib
import json
from typing import List, Dict, Any, Set, Optional
from utils.cache import get_cache_manager
from datetime import datetime

class IncrementalActivityProcessor:
    """
    Handles incremental processing of deal activities.
    Tracks processed activities using hash-based IDs and stores state in cache.
    Only new or updated activities are processed; merges results with previous state.
    """
    def __init__(self, cache_ttl: int = 86400):
        self.cache = get_cache_manager()
        self.cache_ttl = cache_ttl

    def _activity_hash(self, activity: Dict[str, Any]) -> str:
        """Generate a unique hash for an activity based on its content."""
        # Use a stable, order-independent hash (sort keys)
        activity_json = json.dumps(activity, sort_keys=True, default=str)
        return hashlib.sha256(activity_json.encode('utf-8')).hexdigest()

    def get_processed_activity_ids(self, deal_id: str) -> Set[str]:
        """Retrieve the set of processed activity hashes for a deal from cache."""
        key = f"deal_activities:{deal_id}"
        ids = self.cache.get(key, default=[], cache_type="incremental")
        return set(ids)

    def set_processed_activity_ids(self, deal_id: str, activity_ids: Set[str]):
        """Store the set of processed activity hashes for a deal in cache."""
        key = f"deal_activities:{deal_id}"
        self.cache.set(key, list(activity_ids), ttl=self.cache_ttl, cache_type="incremental")

    def detect_new_or_updated_activities(self, deal_id: str, activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return only new or updated activities for a deal."""
        processed_ids = self.get_processed_activity_ids(deal_id)
        new_activities = []
        for activity in activities:
            act_hash = self._activity_hash(activity)
            if act_hash not in processed_ids:
                new_activities.append(activity)
        return new_activities

    def update_state_after_processing(self, deal_id: str, activities: List[Dict[str, Any]]):
        """Update the cache with hashes of all activities that have now been processed."""
        all_ids = {self._activity_hash(a) for a in activities}
        self.set_processed_activity_ids(deal_id, all_ids)

    def merge_results(self, previous_result: Optional[Dict[str, Any]], new_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge previous and new analysis results. This is a naive merge; customize as needed.
        - For lists (e.g., activities), concatenate and deduplicate.
        - For dicts, update with new values.
        - For metrics, recalculate or update as needed.
        """
        if not previous_result:
            return new_result
        merged = previous_result.copy()
        for k, v in new_result.items():
            if isinstance(v, list) and k in merged:
                # Merge and deduplicate by hash if possible
                seen = set()
                merged_list = []
                for item in merged[k] + v:
                    item_hash = hashlib.sha256(json.dumps(item, sort_keys=True, default=str).encode('utf-8')).hexdigest()
                    if item_hash not in seen:
                        seen.add(item_hash)
                        merged_list.append(item)
                merged[k] = merged_list
            elif isinstance(v, dict) and k in merged:
                merged[k].update(v)
            else:
                merged[k] = v
        return merged

    def clear_state(self, deal_id: str):
        """Remove cached state for a deal (for invalidation)."""
        key = f"deal_activities:{deal_id}"
        self.cache.delete(key, cache_type="incremental")
