"""Operational monitoring of the Somalia riverine flood trigger.

Daily pipeline (pipelines/): fetch the operational forecasts at the seven
trigger points, evaluate the four river-season windows, chart, email via
Listmonk and publish a status snapshot for the public dashboard.
The trigger definition itself lives in src.constants (TRIGGER_CONFIG); this
package only applies it.
"""
