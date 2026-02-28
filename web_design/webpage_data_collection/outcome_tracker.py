"""
Outcome Tracker
Integrates with analytics platforms to measure webpage outcomes

Supports:
- Google Analytics 4 (GA4)
- Mixpanel
- Segment
- PostHog  
- Custom analytics endpoints

Measures:
- Conversion rates (primary outcome)
- Engagement metrics (time on page, scroll depth, clicks)
- Bounce/exit rates
- A/B test results
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from .state_schema import OutcomeMetrics


class OutcomeTracker:
    """
    Base class for tracking webpage outcomes from analytics platforms
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def measure_outcomes(
        self,
        site_id: str,
        page_url: str,
        start_time: datetime,
        end_time: datetime,
        conversion_goal: str = "signup"
    ) -> OutcomeMetrics:
        """
        Measure outcomes for a specific page over a time period
        
        Args:
            site_id: Site identifier
            page_url: URL of the page to measure
            start_time: Start of measurement period
            end_time: End of measurement period
            conversion_goal: What counts as a conversion
        
        Returns:
            OutcomeMetrics object with all measured outcomes
        """
        raise NotImplementedError("Subclasses must implement measure_outcomes")
    
    def measure_ab_test(
        self,
        site_id: str,
        page_url: str,
        variant_a_state_id: str,
        variant_b_state_id: str,
        duration_hours: float = 24.0,
        min_visitors_per_variant: int = 100
    ) -> Dict[str, float]:
        """
        Run A/B test comparing two page variants
        
        Args:
            site_id: Site identifier
            page_url: URL being tested
            variant_a_state_id: ID of control state
            variant_b_state_id: ID of treatment state
            duration_hours: How long to run test
            min_visitors_per_variant: Minimum sample size
        
        Returns:
            Dict with statistical comparison results
        """
        # Measure both variants
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=duration_hours)
        
        outcomes_a = self.measure_outcomes(site_id, page_url, start_time, end_time)
        outcomes_b = self.measure_outcomes(site_id, page_url, start_time, end_time)
        
        # Compute deltas
        results = {
            'conversion_rate_delta': (outcomes_b.conversion_rate or 0) - (outcomes_a.conversion_rate or 0),
            'conversion_rate_lift': ((outcomes_b.conversion_rate or 0) / max(outcomes_a.conversion_rate or 0.01, 0.01) - 1) * 100,
            'time_on_page_delta': (outcomes_b.avg_time_on_page or 0) - (outcomes_a.avg_time_on_page or 0),
            'bounce_rate_delta': (outcomes_b.bounce_rate or 0) - (outcomes_a.bounce_rate or 0),
            'variant_a_visitors': outcomes_a.unique_visitors or 0,
            'variant_b_visitors': outcomes_b.unique_visitors or 0,
        }
        
        # Statistical significance (placeholder - would use proper t-test)
        results['statistically_significant'] = abs(results['conversion_rate_delta']) > 0.05
        results['confidence_level'] = 0.95 if results['statistically_significant'] else 0.70
        
        return results


class GA4OutcomeTracker(OutcomeTracker):
    """
    Google Analytics 4 integration
    
    Uses GA4 Data API v1 to query metrics
    """
    
    def __init__(self, property_id: str, credentials_path: str):
        """
        Args:
            property_id: GA4 property ID (e.g., "123456789")
            credentials_path: Path to service account JSON
        """
        super().__init__()
        self.property_id = property_id
        self.credentials_path = credentials_path
        
        # In production, would initialize Google Analytics Data API client
        # from google.analytics.data import BetaAnalyticsDataClient
        # self.client = BetaAnalyticsDataClient.from_service_account_json(credentials_path)
    
    def measure_outcomes(
        self,
        site_id: str,
        page_url: str,
        start_time: datetime,
        end_time: datetime,
        conversion_goal: str = "signup"
    ) -> OutcomeMetrics:
        """
        Query GA4 for outcome metrics
        
        Queries:
        - Conversion events (custom event named after conversion_goal)
        - Session metrics (avg session duration, bounce rate)
        - Page metrics (pageviews, unique pageviews)
        - Engagement metrics (scroll depth, click events)
        """
        # Format dates for GA4 API
        start_date = start_time.strftime("%Y-%m-%d")
        end_date = end_time.strftime("%Y-%m-%d")
        
        # Build dimension filter for specific page
        page_filter = self._build_page_filter(page_url)
        
        # Query conversion metrics
        conversion_data = self._query_ga4_conversions(
            start_date, end_date, page_filter, conversion_goal
        )
        
        # Query engagement metrics
        engagement_data = self._query_ga4_engagement(
            start_date, end_date, page_filter
        )
        
        # Query traffic metrics
        traffic_data = self._query_ga4_traffic(
            start_date, end_date, page_filter
        )
        
        # Compute duration in hours
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        # Build OutcomeMetrics object
        return OutcomeMetrics(
            conversion_rate=conversion_data.get('conversion_rate'),
            conversion_count=conversion_data.get('conversion_count'),
            conversion_goal=conversion_goal,
            avg_time_on_page=engagement_data.get('avg_session_duration'),
            scroll_depth_avg=engagement_data.get('avg_scroll_depth'),
            clicks_per_session=engagement_data.get('clicks_per_session'),
            bounce_rate=engagement_data.get('bounce_rate'),
            exit_rate=engagement_data.get('exit_rate'),
            unique_visitors=traffic_data.get('users'),
            pageviews=traffic_data.get('pageviews'),
            measurement_start=start_date,
            measurement_end=end_date,
            measurement_duration_hours=duration_hours,
        )
    
    def _build_page_filter(self, page_url: str) -> Dict:
        """Build dimension filter for specific page URL"""
        return {
            'filter': {
                'fieldName': 'pagePath',
                'stringFilter': {
                    'matchType': 'EXACT',
                    'value': page_url
                }
            }
        }
    
    def _query_ga4_conversions(
        self,
        start_date: str,
        end_date: str,
        page_filter: Dict,
        conversion_goal: str
    ) -> Dict:
        """
        Query GA4 for conversion metrics
        
        Metrics:
        - conversions (count of conversion_goal event)
        - sessions (total sessions)
        - conversion_rate = conversions / sessions
        """
        # In production:
        # request = RunReportRequest(
        #     property=f"properties/{self.property_id}",
        #     date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        #     metrics=[
        #         Metric(name="conversions"),
        #         Metric(name="sessions"),
        #     ],
        #     dimensions=[Dimension(name="eventName")],
        #     dimension_filter=FilterExpression(filter=Filter(
        #         field_name="eventName",
        #         string_filter=StringFilter(value=conversion_goal)
        #     ))
        # )
        # response = self.client.run_report(request)
        
        # Placeholder: Return mock data
        return {
            'conversion_count': 42,
            'sessions': 500,
            'conversion_rate': 0.084,  # 8.4%
        }
    
    def _query_ga4_engagement(
        self,
        start_date: str,
        end_date: str,
        page_filter: Dict
    ) -> Dict:
        """
        Query GA4 for engagement metrics
        
        Metrics:
        - avgSessionDuration
        - scrollDepth (custom metric)
        - eventCount / sessions (clicks per session)
        - bounceRate
        - exitRate
        """
        # In production: Similar to _query_ga4_conversions
        
        # Placeholder: Return mock data
        return {
            'avg_session_duration': 145.3,  # seconds
            'avg_scroll_depth': 0.68,  # 68%
            'clicks_per_session': 3.2,
            'bounce_rate': 0.42,  # 42%
            'exit_rate': 0.35,  # 35%
        }
    
    def _query_ga4_traffic(
        self,
        start_date: str,
        end_date: str,
        page_filter: Dict
    ) -> Dict:
        """
        Query GA4 for traffic metrics
        
        Metrics:
        - activeUsers (unique visitors)
        - screenPageViews (pageviews)
        """
        # In production: Similar to _query_ga4_conversions
        
        # Placeholder: Return mock data
        return {
            'users': 450,
            'pageviews': 687,
        }


class MixpanelOutcomeTracker(OutcomeTracker):
    """
    Mixpanel integration
    
    Uses Mixpanel Query API to get event-based metrics
    """
    
    def __init__(self, project_id: str, api_secret: str):
        """
        Args:
            project_id: Mixpanel project ID
            api_secret: Mixpanel API secret for authentication
        """
        super().__init__(api_secret)
        self.project_id = project_id
        self.api_base = "https://mixpanel.com/api/2.0"
    
    def measure_outcomes(
        self,
        site_id: str,
        page_url: str,
        start_time: datetime,
        end_time: datetime,
        conversion_goal: str = "signup"
    ) -> OutcomeMetrics:
        """
        Query Mixpanel for outcome metrics
        
        Uses segmentation and funnels APIs
        """
        # Format dates for Mixpanel
        from_date = start_time.strftime("%Y-%m-%d")
        to_date = end_time.strftime("%Y-%m-%d")
        
        # Query conversion funnel
        funnel_data = self._query_mixpanel_funnel(
            from_date, to_date, page_url, conversion_goal
        )
        
        # Query engagement events
        engagement_data = self._query_mixpanel_events(
            from_date, to_date, page_url
        )
        
        # Compute metrics
        pageviews = engagement_data.get('page_view_count', 0)
        conversions = funnel_data.get('conversion_count', 0)
        conversion_rate = conversions / max(pageviews, 1)
        
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        return OutcomeMetrics(
            conversion_rate=conversion_rate,
            conversion_count=conversions,
            conversion_goal=conversion_goal,
            avg_time_on_page=engagement_data.get('avg_time_on_page'),
            scroll_depth_avg=engagement_data.get('avg_scroll_depth'),
            clicks_per_session=engagement_data.get('clicks_per_session'),
            bounce_rate=engagement_data.get('bounce_rate'),
            exit_rate=None,
            unique_visitors=engagement_data.get('unique_users'),
            pageviews=pageviews,
            measurement_start=from_date,
            measurement_end=to_date,
            measurement_duration_hours=duration_hours,
        )
    
    def _query_mixpanel_funnel(
        self,
        from_date: str,
        to_date: str,
        page_url: str,
        conversion_goal: str
    ) -> Dict:
        """
        Query Mixpanel funnel API
        
        Funnel steps:
        1. Page View (page_url)
        2. Conversion Event (conversion_goal)
        """
        # In production:
        # params = {
        #     'from_date': from_date,
        #     'to_date': to_date,
        #     'funnel_id': self._get_or_create_funnel(page_url, conversion_goal),
        # }
        # response = requests.get(
        #     f"{self.api_base}/funnels",
        #     params=params,
        #     auth=(self.api_key, '')
        # )
        # data = response.json()
        
        # Placeholder
        return {
            'conversion_count': 38,
            'step_1_count': 520,
            'conversion_rate': 0.073,
        }
    
    def _query_mixpanel_events(
        self,
        from_date: str,
        to_date: str,
        page_url: str
    ) -> Dict:
        """
        Query Mixpanel segmentation API for engagement events
        """
        # In production: Query for events like:
        # - page_view (filtered by page_url)
        # - click_element
        # - scroll
        # - bounce (derived from single-event sessions)
        
        # Placeholder
        return {
            'page_view_count': 520,
            'unique_users': 420,
            'avg_time_on_page': 132.5,
            'avg_scroll_depth': 0.65,
            'clicks_per_session': 2.8,
            'bounce_rate': 0.38,
        }


class SegmentOutcomeTracker(OutcomeTracker):
    """
    Segment integration
    
    Segment acts as a data pipeline to other analytics platforms
    This tracker queries the downstream warehouse (e.g., BigQuery)
    """
    
    def __init__(self, write_key: str, warehouse_client=None):
        """
        Args:
            write_key: Segment write key (for tracking events)
            warehouse_client: Client for querying warehouse (e.g., BigQuery client)
        """
        super().__init__(write_key)
        self.warehouse_client = warehouse_client
    
    def measure_outcomes(
        self,
        site_id: str,
        page_url: str,
        start_time: datetime,
        end_time: datetime,
        conversion_goal: str = "signup"
    ) -> OutcomeMetrics:
        """
        Query data warehouse for Segment-tracked events
        
        Assumes Segment is piping data to BigQuery/Redshift/Snowflake
        """
        # In production: Run SQL queries against warehouse
        # SELECT 
        #   COUNT(DISTINCT user_id) as unique_visitors,
        #   COUNT(*) as pageviews,
        #   SUM(CASE WHEN event = 'conversion_goal' THEN 1 ELSE 0 END) as conversions,
        #   AVG(time_on_page) as avg_time_on_page
        # FROM segment_events
        # WHERE page_url = %s
        #   AND timestamp BETWEEN %s AND %s
        
        # Placeholder
        return OutcomeMetrics(
            conversion_rate=0.091,
            conversion_count=45,
            conversion_goal=conversion_goal,
            avg_time_on_page=138.7,
            scroll_depth_avg=0.71,
            clicks_per_session=3.5,
            bounce_rate=0.40,
            exit_rate=0.33,
            unique_visitors=495,
            pageviews=612,
            measurement_start=start_time.strftime("%Y-%m-%d"),
            measurement_end=end_time.strftime("%Y-%m-%d"),
            measurement_duration_hours=(end_time - start_time).total_seconds() / 3600,
        )


class CustomOutcomeTracker(OutcomeTracker):
    """
    Custom analytics endpoint integration
    
    For sites that have their own analytics API/database
    """
    
    def __init__(self, api_endpoint: str, api_key: str):
        """
        Args:
            api_endpoint: Base URL for custom analytics API
            api_key: Authentication key
        """
        super().__init__(api_key)
        self.api_endpoint = api_endpoint
    
    def measure_outcomes(
        self,
        site_id: str,
        page_url: str,
        start_time: datetime,
        end_time: datetime,
        conversion_goal: str = "signup"
    ) -> OutcomeMetrics:
        """
        Query custom analytics endpoint
        
        Expects endpoint to return JSON with standard metrics
        """
        params = {
            'site_id': site_id,
            'page_url': page_url,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'conversion_goal': conversion_goal,
        }
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        try:
            response = requests.get(
                f"{self.api_endpoint}/outcomes",
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response into OutcomeMetrics
            return OutcomeMetrics(
                conversion_rate=data.get('conversion_rate'),
                conversion_count=data.get('conversion_count'),
                conversion_goal=conversion_goal,
                avg_time_on_page=data.get('avg_time_on_page'),
                scroll_depth_avg=data.get('scroll_depth_avg'),
                clicks_per_session=data.get('clicks_per_session'),
                bounce_rate=data.get('bounce_rate'),
                exit_rate=data.get('exit_rate'),
                unique_visitors=data.get('unique_visitors'),
                pageviews=data.get('pageviews'),
                measurement_start=start_time.strftime("%Y-%m-%d"),
                measurement_end=end_time.strftime("%Y-%m-%d"),
                measurement_duration_hours=data.get('measurement_duration_hours'),
            )
        
        except requests.RequestException as e:
            print(f"Error querying custom analytics endpoint: {e}")
            # Return empty metrics
            return OutcomeMetrics(
                conversion_rate=None,
                conversion_count=None,
                conversion_goal=conversion_goal,
                measurement_start=start_time.strftime("%Y-%m-%d"),
                measurement_end=end_time.strftime("%Y-%m-%d"),
                measurement_duration_hours=(end_time - start_time).total_seconds() / 3600,
            )


def create_outcome_tracker(
    platform: str,
    **kwargs
) -> OutcomeTracker:
    """
    Factory function to create appropriate outcome tracker
    
    Args:
        platform: Analytics platform ("ga4", "mixpanel", "segment", "custom")
        **kwargs: Platform-specific initialization parameters
    
    Returns:
        OutcomeTracker instance
    
    Examples:
        >>> tracker = create_outcome_tracker(
        ...     "ga4",
        ...     property_id="123456789",
        ...     credentials_path="./ga4_credentials.json"
        ... )
        >>> 
        >>> tracker = create_outcome_tracker(
        ...     "mixpanel",
        ...     project_id="abc123",
        ...     api_secret="secret_key"
        ... )
    """
    if platform == "ga4":
        return GA4OutcomeTracker(**kwargs)
    elif platform == "mixpanel":
        return MixpanelOutcomeTracker(**kwargs)
    elif platform == "segment":
        return SegmentOutcomeTracker(**kwargs)
    elif platform == "custom":
        return CustomOutcomeTracker(**kwargs)
    else:
        raise ValueError(f"Unknown platform: {platform}")
