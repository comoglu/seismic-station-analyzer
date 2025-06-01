#!/usr/bin/python3
"""
Seismic Station Noise Analyzer
==============================

A comprehensive GUI application for analyzing seismic station noise levels and evaluating 
station suitability for local earthquake detection. This tool processes seismic data from 
both local files and FDSN web services, calculates power spectral density (PSD), compares 
noise levels against Peterson's New Low/High Noise Models (NLNM/NHNM), and generates 
detailed interactive reports.

Features:
---------
• Load data from MiniSEED files or fetch from FDSN web services
• Advanced PSD calculation with robust statistical methods
• Event detection and removal using STA/LTA algorithms
• Multiple analysis methods (absolute threshold, relative ranking, statistical)
• Compliance checking against standard noise models
• Interactive HTML reports with embedded plots
• Station quality assessment and ranking
• Configurable frequency ranges and analysis parameters

Author: Mustafa Comoglu
Email: comoglu@gmail.com
GitHub: https://github.com/comoglu/seismic-station-analyzer
License: MIT
Version: 1.0.0
Created: 2025

Dependencies:
------------
• obspy >= 1.3.0
• numpy >= 1.20.0
• pandas >= 1.3.0
• scipy >= 1.7.0
• matplotlib >= 3.5.0
• PyQt5 >= 5.15.0

Usage:
------
python seismic-station-analyzer.py

For documentation and examples, visit:
https://github.com/comoglu/seismic-station-analyzer

Citation:
---------
If you use this software in your research, please cite:
Comoglu, M. (2025). Seismic Station Noise Analyzer: A tool for evaluating station 
suitability for local earthquake detection. GitHub repository.
"""

import sys
import os
from pathlib import Path
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import numpy as np
import pandas as pd
import datetime
from scipy import signal
import base64
from io import BytesIO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QTabWidget, QComboBox,
                             QLineEdit, QCheckBox, QProgressBar, QTableWidget, QTableWidgetItem,
                             QSplitter, QMessageBox, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
                             QButtonGroup,QRadioButton,QStackedWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import obspy
from obspy import read, read_inventory
from obspy.signal.spectral_estimation import get_nlnm, get_nhnm
from obspy.clients.fdsn import Client

# Important: Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to Qt5Agg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class StationNoiseAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seismic Station Noise Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Data storage
        self.miniseed_file = None
        self.inventory_file = None
        self.stream = None
        self.inventory = None
        self.results_df = None
        self.station_plots = {}  # Cache for station plots
        self.overlap = QDoubleSpinBox()  # Initialize overlap attribute
        self.use_weighting = QCheckBox()  # Initialize use_weighting attribute

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create tabs for different functionalities
        self.setup_data_tab()
        self.setup_analysis_tab()
        self.setup_results_tab()
        self.setup_plots_tab()

        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

    def setup_data_tab(self):
        # Create widget and layout for the data tab
        data_widget = QWidget()
        data_layout = QVBoxLayout(data_widget)

        # File input section
        file_group = QGroupBox("Local Files")
        file_layout = QFormLayout()

        # Miniseed file selection
        miniseed_layout = QHBoxLayout()
        self.miniseed_path = QLineEdit()
        self.miniseed_path.setReadOnly(True)
        miniseed_button = QPushButton("Browse...")
        miniseed_button.clicked.connect(self.select_miniseed)
        miniseed_layout.addWidget(self.miniseed_path)
        miniseed_layout.addWidget(miniseed_button)
        file_layout.addRow("Miniseed File:", miniseed_layout)

        # Inventory file selection
        inventory_layout = QHBoxLayout()
        self.inventory_path = QLineEdit()
        self.inventory_path.setReadOnly(True)
        inventory_button = QPushButton("Browse...")
        inventory_button.clicked.connect(self.select_inventory)
        inventory_layout.addWidget(self.inventory_path)
        inventory_layout.addWidget(inventory_button)
        file_layout.addRow("Inventory (Optional):", inventory_layout)

        # Load button
        load_button = QPushButton("Load Data")
        load_button.clicked.connect(self.load_data)
        file_layout.addWidget(load_button)

        file_group.setLayout(file_layout)
        data_layout.addWidget(file_group)

        # FDSN section
        fdsn_group = QGroupBox("Fetch from FDSN Web Service")
        fdsn_layout = QFormLayout()

        # FDSN Server selection
        self.fdsn_server = QComboBox()
        self.fdsn_server.addItems(["IRIS", "AUSPASS","RASPISHAKE", "GEOFON", "KOERI", "USGS", "INGV", "NCEDC", "RESIF", "Custom"])
        fdsn_layout.addRow("Server:", self.fdsn_server)

        # Custom server URL
        self.custom_server_url = QLineEdit()
        self.custom_server_url.setEnabled(False)
        self.fdsn_server.currentTextChanged.connect(self.toggle_custom_server)
        fdsn_layout.addRow("Custom URL:", self.custom_server_url)

        # Network, Station, Location, Channel
        self.network = QLineEdit("*")
        fdsn_layout.addRow("Network:", self.network)

        self.station = QLineEdit("*")
        fdsn_layout.addRow("Station:", self.station)

        self.location = QLineEdit("*")
        fdsn_layout.addRow("Location:", self.location)

        self.channel = QLineEdit("?H?")
        fdsn_layout.addRow("Channel:", self.channel)

        # Time range with more detailed format
        # Use current date and time as default with UTC awareness
        current_datetime = datetime.datetime.now(datetime.UTC)
        default_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        self.start_time = QLineEdit(default_datetime)
        # Add placeholder text to show format
        self.start_time.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        fdsn_layout.addRow("Start Time (UTC):", self.start_time)

        # Add a tooltip to explain the format
        self.start_time.setToolTip(
            "Enter the start time in format: YYYY-MM-DD HH:MM:SS\n"
            "Example: 2025-03-06 08:30:00 for March 6, 2025 at 8:30 AM UTC"
        )

        self.duration = QSpinBox()
        self.duration.setRange(1, 24)
        self.duration.setValue(1)
        fdsn_layout.addRow("Duration (hours):", self.duration)

        # Fetch button
        fetch_button = QPushButton("Fetch Data")
        fetch_button.clicked.connect(self.fetch_data)
        fdsn_layout.addWidget(fetch_button)

        fdsn_group.setLayout(fdsn_layout)
        data_layout.addWidget(fdsn_group)

        # Add data_widget to the tabs
        self.tabs.addTab(data_widget, "Data Source")

        # Data overview section
        self.data_overview_group = QGroupBox("Data Overview")
        data_overview_layout = QFormLayout()

        self.data_timespan = QLabel("No data loaded")
        data_overview_layout.addRow("Time Span:", self.data_timespan)

        self.data_stations = QLabel("0 stations")
        data_overview_layout.addRow("Stations:", self.data_stations)

        self.data_channels = QLabel("0 channels")
        data_overview_layout.addRow("Channels:", self.data_channels)

        self.data_quality = QLabel("N/A")
        data_overview_layout.addRow("Overall Quality:", self.data_quality)

        self.data_overview_group.setLayout(data_overview_layout)
        data_layout.addWidget(self.data_overview_group)

    def update_data_overview(self):
        """Update the data overview section with current data stats"""
        if not self.stream:
            return
        
        # Calculate time span
        start_times = [tr.stats.starttime for tr in self.stream]
        end_times = [tr.stats.endtime for tr in self.stream]
        
        earliest = min(start_times)
        latest = max(end_times)
        duration_hours = (latest - earliest) / 3600
        
        self.data_timespan.setText(
            f"{earliest.strftime('%Y-%m-%d %H:%M')} to {latest.strftime('%Y-%m-%d %H:%M')} "
            f"({duration_hours:.1f} hours)"
        )
        
        # Count unique stations and channels
        stations = set()
        channels = set()
        
        for tr in self.stream:
            stations.add(f"{tr.stats.network}.{tr.stats.station}")
            channels.add(tr.stats.channel)
        
        self.data_stations.setText(f"{len(stations)} stations")
        self.data_channels.setText(f"{len(channels)} channels")
        
        # Basic quality assessment
        has_issues = False
        issue_count = 0
        
        for tr in self.stream:
            quality = self.check_data_quality(tr)
            if quality['issues']:
                has_issues = True
                issue_count += len(quality['issues'])
        
        if has_issues:
            self.data_quality.setText(f"Issues detected ({issue_count})")
            self.data_quality.setStyleSheet("color: orange")
        else:
            self.data_quality.setText("Good")
            self.data_quality.setStyleSheet("color: green")

    def update_analysis_options(self):
        """Update UI based on the selected analysis method"""
        if self.absolute_threshold.isChecked():
            self.method_options_stack.setCurrentIndex(0)
        elif self.relative_ranking.isChecked():
            self.method_options_stack.setCurrentIndex(1)
        elif self.statistical_method.isChecked():
            self.method_options_stack.setCurrentIndex(2)

    def apply_freq_preset(self, preset_text):
        """Apply predefined frequency range presets"""
        # Store the current preset name
        self.current_preset_name = preset_text
        
        if preset_text == "Teleseismic (0.1-1 Hz)":
            self.min_freq.setValue(0.1)
            self.max_freq.setValue(1.0)
        elif preset_text == "Regional (1-10 Hz)":
            self.min_freq.setValue(1.0)
            self.max_freq.setValue(10.0)
        elif preset_text == "Local (1-8 Hz)":  # Changed from "Local (10-40 Hz)"
            self.min_freq.setValue(1.0)        # Changed from 10.0
            self.max_freq.setValue(8.0)        # Changed from 40.0
        # For "Custom", do nothing and let user set values

    def check_data_quality(self, trace):
        """
        Check trace data quality and return quality metrics.
        
        Parameters:
        trace (obspy.Trace): Seismic trace to check
        
        Returns:
        dict: Dictionary of quality metrics
        """
        # Initialize quality metrics
        quality = {
            'has_gaps': False,
            'gap_percentage': 0.0,
            'has_spikes': False,
            'has_flatlines': False,
            'completeness': 100.0,
            'quality_score': 10.0,  # Scale 0-10, 10 being best
            'issues': []
        }
        
        # Check for gaps (we're approximating here, actual gap detection would need trace merging info)
        if hasattr(trace.stats, 'gap_count') and trace.stats.gap_count > 0:
            quality['has_gaps'] = True
            quality['gap_percentage'] = min(100.0, trace.stats.gap_count * 100.0 / len(trace.data))
            quality['issues'].append(f"Found {trace.stats.gap_count} data gaps")
            quality['quality_score'] -= min(3.0, quality['gap_percentage'] / 10.0)
        
        # Check for spikes (values exceeding 5 standard deviations)
        data_mean = np.mean(trace.data)
        data_std = np.std(trace.data)
        spike_threshold = data_std * 5.0
        spike_count = np.sum(np.abs(trace.data - data_mean) > spike_threshold)
        spike_percentage = spike_count * 100.0 / len(trace.data)
        
        if spike_percentage > 0.1:  # More than 0.1% of data are spikes
            quality['has_spikes'] = True
            quality['issues'].append(f"Found potential spikes ({spike_percentage:.2f}% of data)")
            quality['quality_score'] -= min(2.0, spike_percentage)
        
        # Check for flatlines (constant values for extended periods)
        diff = np.diff(trace.data)
        zero_diff_percentage = np.sum(diff == 0) * 100.0 / len(diff)
        
        if zero_diff_percentage > 1.0:  # More than 1% of data are flatlines
            quality['has_flatlines'] = True
            quality['issues'].append(f"Found potential flatlines ({zero_diff_percentage:.2f}% of data)")
            quality['quality_score'] -= min(2.0, zero_diff_percentage / 5.0)
        
        # Calculate completeness (approximation based on issues found)
        deduction = (quality['gap_percentage'] + spike_percentage + zero_diff_percentage) / 2.0
        quality['completeness'] = max(0.0, 100.0 - deduction)
        
        # Ensure quality score stays within 0-10 range
        quality['quality_score'] = max(0.0, min(10.0, quality['quality_score']))
        
        return quality

    def setup_analysis_tab(self):
        # Create widget and layout for the analysis tab
        analysis_widget = QWidget()
        analysis_layout = QVBoxLayout(analysis_widget)
        
        # Parameters section
        params_group = QGroupBox("Frequency Range Parameters")
        params_layout = QFormLayout()
        
        # Local seismicity frequency range
        self.min_freq = QDoubleSpinBox()
        self.min_freq.setRange(0.1, 50)
        self.min_freq.setValue(1.0)
        self.min_freq.setDecimals(1)
        params_layout.addRow("Min Frequency (Hz):", self.min_freq)
        
        self.max_freq = QDoubleSpinBox()
        self.max_freq.setRange(0.1, 50)
        self.max_freq.setValue(8.0)
        self.max_freq.setDecimals(1)
        params_layout.addRow("Max Frequency (Hz):", self.max_freq)

        # Add frequency range presets
        presets_layout = QHBoxLayout()
        presets_layout.addWidget(QLabel("Frequency Presets:"))

        self.freq_presets = QComboBox()
        self.freq_presets.addItems(["Custom", "Teleseismic (0.7-2 Hz)", "Regional (1-10 Hz)", "Local (1-8 Hz)"])
        self.freq_presets.currentTextChanged.connect(self.apply_freq_preset)
        presets_layout.addWidget(self.freq_presets)

        params_layout.addRow(presets_layout)

        params_group.setLayout(params_layout)
        analysis_layout.addWidget(params_group)
        
        # Analysis method section
        analysis_method_group = QGroupBox("Analysis Method")
        analysis_method_layout = QVBoxLayout()
        
        # Radio buttons for different methods
        self.analysis_method_buttons = QButtonGroup()
        self.absolute_threshold = QRadioButton("Absolute Threshold")
        self.relative_ranking = QRadioButton("Relative Ranking")
        self.statistical_method = QRadioButton("Statistical Method")
        
        self.analysis_method_buttons.addButton(self.absolute_threshold)
        self.analysis_method_buttons.addButton(self.relative_ranking)
        self.analysis_method_buttons.addButton(self.statistical_method)
        
        self.absolute_threshold.setChecked(True)
        analysis_method_layout.addWidget(self.absolute_threshold)
        analysis_method_layout.addWidget(self.relative_ranking)
        analysis_method_layout.addWidget(self.statistical_method)

        # Add tooltips for analysis methods
        self.absolute_threshold.setToolTip(
            "Sets a fixed threshold (in dB) for how much a station's noise can exceed the NLNM. "
            "Stations with noise levels below this threshold are considered suitable."
        )
        self.relative_ranking.setToolTip(
            "Ranks stations based on how far they are above the NLNM, then selects a percentage of the best stations. "
            "Useful when you need a specific number of stations regardless of absolute noise levels."
        )
        self.statistical_method.setToolTip(
            "Identifies stations within a certain number of standard deviations from the best station. "
            "Useful for finding stations that are statistically similar to your best station."
        )

        # Add tooltips for PSD parameters
        self.segment_length = QDoubleSpinBox()
        self.segment_length.setToolTip(
            "Length of each segment for PSD calculation in seconds. "
            "Longer segments give better frequency resolution but might miss temporal changes."
        )
        self.overlap.setToolTip(
            "Overlap between consecutive segments (0-1). "
            "Higher values give smoother results but introduce more correlation between segments."
        )
        self.use_weighting.setToolTip(
            "Apply frequency-dependent weighting to emphasize certain frequency bands. "
            "Useful when specific frequencies are more important for your target signals."
        )
        
        # Connect signals to update UI based on selection
        self.absolute_threshold.toggled.connect(self.update_analysis_options)
        self.relative_ranking.toggled.connect(self.update_analysis_options)
        self.statistical_method.toggled.connect(self.update_analysis_options)
        
        # Stacked widget for method-specific options
        self.method_options_stack = QStackedWidget()
        
        # 1. Absolute threshold options
        absolute_widget = QWidget()
        absolute_layout = QFormLayout(absolute_widget)
        
        self.noise_threshold = QDoubleSpinBox()
        self.noise_threshold.setRange(10, 200)
        self.noise_threshold.setValue(60.0)  # A more reasonable default value
        self.noise_threshold.setDecimals(1)
        absolute_layout.addRow("Noise Level Threshold (dB above NLNM):", self.noise_threshold)
        
        self.method_options_stack.addWidget(absolute_widget)
        
        # 2. Relative ranking options
        relative_widget = QWidget()
        relative_layout = QFormLayout(relative_widget)
        
        self.percentile_threshold = QSpinBox()
        self.percentile_threshold.setRange(10, 90)
        self.percentile_threshold.setValue(30)  # Select top 30% of stations
        relative_layout.addRow("Select top percentage:", self.percentile_threshold)
        
        self.method_options_stack.addWidget(relative_widget)
        
        # 3. Statistical method options
        statistical_widget = QWidget()
        statistical_layout = QFormLayout(statistical_widget)
        
        self.std_dev_threshold = QDoubleSpinBox()
        self.std_dev_threshold.setRange(0.5, 5.0)
        self.std_dev_threshold.setValue(1.5)
        self.std_dev_threshold.setDecimals(1)
        statistical_layout.addRow("Standard deviations from best:", self.std_dev_threshold)
        
        self.method_options_stack.addWidget(statistical_widget)
        
        # Add method options stack to layout
        analysis_method_layout.addWidget(self.method_options_stack)
        analysis_method_group.setLayout(analysis_method_layout)
        analysis_layout.addWidget(analysis_method_group)

        # Add a section for event detection parameters
        event_group = QGroupBox("Event Detection Parameters")
        event_layout = QFormLayout()

        self.sta_length = QDoubleSpinBox()
        self.sta_length.setRange(1.0, 10.0)
        self.sta_length.setValue(3.0)  # Default 3s STA window
        self.sta_length.setDecimals(1)
        event_layout.addRow("STA window length (s):", self.sta_length)

        self.lta_length = QDoubleSpinBox()
        self.lta_length.setRange(5.0, 60.0)
        self.lta_length.setValue(30.0)  # Default 30s LTA window
        self.lta_length.setDecimals(1)
        event_layout.addRow("LTA window length (s):", self.lta_length)

        self.trigger_threshold = QDoubleSpinBox()
        self.trigger_threshold.setRange(1.5, 10.0)
        self.trigger_threshold.setValue(3.5)  # Default threshold
        self.trigger_threshold.setDecimals(1)
        event_layout.addRow("Trigger threshold:", self.trigger_threshold)

        self.event_buffer = QDoubleSpinBox()
        self.event_buffer.setRange(0.0, 30.0)
        self.event_buffer.setValue(10.0)  # Default 10s buffer
        self.event_buffer.setDecimals(1)
        event_layout.addRow("Buffer around events (s):", self.event_buffer)

        self.use_event_detection = QCheckBox()
        self.use_event_detection.setChecked(True)
        event_layout.addRow("Remove events from analysis:", self.use_event_detection)

        event_group.setLayout(event_layout)
        analysis_layout.addWidget(event_group)

        # Advanced PSD calculation parameters section
        psd_group = QGroupBox("PSD Calculation Parameters")
        psd_layout = QFormLayout()
        
        # PSD parameters
        self.segment_length = QDoubleSpinBox()
        self.segment_length.setRange(10, 900)
        self.segment_length.setValue(90.0)
        self.segment_length.setDecimals(1)
        psd_layout.addRow("Segment Length (s):", self.segment_length)
        
        self.overlap = QDoubleSpinBox()
        self.overlap.setRange(0, 0.95)
        self.overlap.setValue(0.5)
        self.overlap.setDecimals(2)
        psd_layout.addRow("Overlap:", self.overlap)
        
        # Add frequency weighting option
        self.use_weighting = QCheckBox()
        self.use_weighting.setChecked(True)
        psd_layout.addRow("Apply frequency weighting:", self.use_weighting)
        
        self.peak_weight_freq = QDoubleSpinBox()
        self.peak_weight_freq.setRange(0.5, 25)
        self.peak_weight_freq.setValue(10.0)
        self.peak_weight_freq.setDecimals(1)
        psd_layout.addRow("Peak weight frequency (Hz):", self.peak_weight_freq)
        
        psd_group.setLayout(psd_layout)
        analysis_layout.addWidget(psd_group)
        
        # Analysis button
        analyze_button = QPushButton("Run Analysis")
        analyze_button.clicked.connect(self.run_analysis)
        analysis_layout.addWidget(analyze_button)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QFormLayout()
        
        self.output_dir = QLineEdit("noise_report")
        output_dir_button = QPushButton("Browse...")
        output_dir_button.clicked.connect(self.select_output_dir)
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir)
        output_dir_layout.addWidget(output_dir_button)
        output_layout.addRow("Output Directory:", output_dir_layout)
        
        self.save_plots = QCheckBox()
        self.save_plots.setChecked(True)
        output_layout.addRow("Save PSD Plots:", self.save_plots)
        
        self.save_html = QCheckBox()
        self.save_html.setChecked(True)
        output_layout.addRow("Save HTML Report:", self.save_html)
        
        self.save_csv = QCheckBox()
        self.save_csv.setChecked(True)
        output_layout.addRow("Save CSV Results:", self.save_csv)
        
        # Report button
        export_button = QPushButton("Export Report")
        export_button.clicked.connect(self.export_report)
        output_layout.addWidget(export_button)
        
        output_group.setLayout(output_layout)
        analysis_layout.addWidget(output_group)
        
        # Add analysis_widget to the tabs
        self.tabs.addTab(analysis_widget, "Analysis")
        
        # Set initial state
        self.update_analysis_options()

        # Settings save/load group
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QHBoxLayout()

        self.settings_name = QLineEdit("Default")
        settings_layout.addWidget(QLabel("Name:"))
        settings_layout.addWidget(self.settings_name)

        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.clicked.connect(self.save_analysis_settings)
        settings_layout.addWidget(save_settings_btn)

        self.load_settings_combo = QComboBox()
        self.load_settings_combo.addItem("Default")
        settings_layout.addWidget(self.load_settings_combo)

        load_settings_btn = QPushButton("Load Settings")
        load_settings_btn.clicked.connect(self.load_analysis_settings)
        settings_layout.addWidget(load_settings_btn)

        settings_group.setLayout(settings_layout)
        analysis_layout.addWidget(settings_group)

    def save_analysis_settings(self):
        """Save current analysis settings"""
        settings_name = self.settings_name.text().strip()
        if not settings_name:
            QMessageBox.warning(self, "Warning", "Please enter a name for the settings.")
            return
        
        # Collect all settings
        settings = {
            'min_freq': self.min_freq.value(),
            'max_freq': self.max_freq.value(),
            'segment_length': self.segment_length.value(),
            'overlap': self.overlap.value(),
            'use_weighting': self.use_weighting.isChecked(),
            'peak_weight_freq': self.peak_weight_freq.value(),
            'analysis_method': 'absolute' if self.absolute_threshold.isChecked() else 
                            ('relative' if self.relative_ranking.isChecked() else 'statistical'),
            'noise_threshold': self.noise_threshold.value(),
            'percentile_threshold': self.percentile_threshold.value(),
            'std_dev_threshold': self.std_dev_threshold.value()
        }
        
        # Save settings to a file
        settings_dir = Path("settings")
        settings_dir.mkdir(exist_ok=True)
        
        settings_file = settings_dir / f"{settings_name}.json"
        
        with open(settings_file, 'w') as f:
            import json
            json.dump(settings, f, indent=2)
        
        # Add to combo box if not already there
        if self.load_settings_combo.findText(settings_name) == -1:
            self.load_settings_combo.addItem(settings_name)
        
        self.status_bar.showMessage(f"Settings saved as '{settings_name}'")

    def load_analysis_settings(self):
        """Load saved analysis settings"""
        settings_name = self.load_settings_combo.currentText()
        if settings_name == "Default":
            # Reset to default values
            self.min_freq.setValue(1.0)
            self.max_freq.setValue(20.0)
            self.segment_length.setValue(60.0)
            self.overlap.setValue(0.5)
            self.use_weighting.setChecked(True)
            self.peak_weight_freq.setValue(10.0)
            self.absolute_threshold.setChecked(True)
            self.noise_threshold.setValue(60.0)
            self.percentile_threshold.setValue(30)
            self.std_dev_threshold.setValue(1.5)
            self.status_bar.showMessage("Default settings loaded")
            return
        
        # Load settings from file
        settings_file = Path("settings") / f"{settings_name}.json"
        
        if not settings_file.exists():
            QMessageBox.warning(self, "Warning", f"Settings file '{settings_name}' not found.")
            return
        
        try:
            with open(settings_file, 'r') as f:
                import json
                settings = json.load(f)
            
            # Apply settings
            self.min_freq.setValue(settings.get('min_freq', 1.0))
            self.max_freq.setValue(settings.get('max_freq', 20.0))
            self.segment_length.setValue(settings.get('segment_length', 60.0))
            self.overlap.setValue(settings.get('overlap', 0.5))
            self.use_weighting.setChecked(settings.get('use_weighting', True))
            self.peak_weight_freq.setValue(settings.get('peak_weight_freq', 10.0))
            
            # Set analysis method
            method = settings.get('analysis_method', 'absolute')
            if method == 'absolute':
                self.absolute_threshold.setChecked(True)
            elif method == 'relative':
                self.relative_ranking.setChecked(True)
            else:
                self.statistical_method.setChecked(True)
            
            self.noise_threshold.setValue(settings.get('noise_threshold', 60.0))
            self.percentile_threshold.setValue(settings.get('percentile_threshold', 30))
            self.std_dev_threshold.setValue(settings.get('std_dev_threshold', 1.5))
            
            self.status_bar.showMessage(f"Settings loaded from '{settings_name}'")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load settings: {str(e)}")

    def populate_settings_combo(self):
        """Find and populate the settings combo box with saved settings"""
        settings_dir = Path("settings")
        if not settings_dir.exists():
            return
        
        for settings_file in settings_dir.glob("*.json"):
            settings_name = settings_file.stem
            if self.load_settings_combo.findText(settings_name) == -1:
                self.load_settings_combo.addItem(settings_name)

    def setup_results_tab(self):
        # Create widget and layout for the results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Station", "Channel", "Mean Power (dB)", "SNR to NLNM (dB)", "Suitable", "Duration (hours)"
        ])
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        results_layout.addWidget(self.results_table)

        # Add results_widget to the tabs
        self.tabs.addTab(results_widget, "Results")

    def setup_plots_tab(self):
        # Create widget and layout for the plots tab
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)

        # Station selector
        station_layout = QHBoxLayout()
        station_layout.addWidget(QLabel("Select Station:"))
        self.station_selector = QComboBox()
        self.station_selector.currentTextChanged.connect(self.update_plot)
        station_layout.addWidget(self.station_selector)
        plots_layout.addLayout(station_layout)

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        plots_layout.addWidget(self.canvas)

        # Add plots_widget to the tabs
        self.tabs.addTab(plots_widget, "Plots")

    def select_miniseed(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Miniseed File", "", "MiniSEED Files (*.mseed *.miniseed *.ms);;All Files (*)"
        )
        if file_path:
            self.miniseed_path.setText(file_path)

    def select_inventory(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Inventory File", "", "StationXML Files (*.xml);;All Files (*)"
        )
        if file_path:
            self.inventory_path.setText(file_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        if dir_path:
            self.output_dir.setText(dir_path)

    def toggle_custom_server(self, text):
        self.custom_server_url.setEnabled(text == "Custom")

    def create_quiet_mask(self, trace, triggers):
        """
        Create a boolean mask where True indicates quiet periods (no events).
        
        Parameters:
        trace (obspy.Trace): Seismic trace
        triggers (list): List of (start, end) sample indices for events
        
        Returns:
        numpy.ndarray: Boolean mask where True = quiet periods
        """
        mask = np.ones(len(trace.data), dtype=bool)
        
        # Mark event periods as False
        for start, end in triggers:
            mask[start:end] = False
        
        return mask

    def check_noise_model_compliance(self, period, power, nlnm_periods, nlnm_power, nhnm_periods, nhnm_power):
        """
        Check if station noise falls within NLNM/NHNM bounds and identify frequency ranges with issues.
        
        Parameters:
        period (numpy.ndarray): Period values in seconds
        power (numpy.ndarray): Power values in dB
        nlnm_periods, nlnm_power: Peterson's New Low Noise Model
        nhnm_periods, nhnm_power: Peterson's New High Noise Model
        
        Returns:
        dict: Dictionary of compliance metrics and issues
        """
        # Interpolate noise models to match our periods
        nlnm_interp = np.interp(period, nlnm_periods, nlnm_power)
        nhnm_interp = np.interp(period, nhnm_periods, nhnm_power)
        
        # Check where power exceeds noise models
        below_nlnm = power < nlnm_interp
        above_nhnm = power > nhnm_interp
        within_models = ~(below_nlnm | above_nhnm)
        
        # Get current frequency range of interest
        min_freq = self.min_freq.value()
        max_freq = self.max_freq.value()
        min_period = 1.0 / max_freq
        max_period = 1.0 / min_freq
        
        # Create mask for the frequency range of interest - exactly matching what's shown in the plot
        interest_mask = (period >= min_period) & (period <= max_period)
        
        # Calculate compliance for full range
        total_points = len(period)
        within_models_pct = np.sum(within_models) * 100.0 / total_points
        below_nlnm_pct = np.sum(below_nlnm) * 100.0 / total_points
        above_nhnm_pct = np.sum(above_nhnm) * 100.0 / total_points
        
        # Calculate compliance within range of interest
        interest_points = np.sum(interest_mask)
        if interest_points > 0:
            interest_within = within_models & interest_mask
            interest_below = below_nlnm & interest_mask
            interest_above = above_nhnm & interest_mask
            
            within_models_interest_pct = np.sum(interest_within) * 100.0 / interest_points
            below_nlnm_interest_pct = np.sum(interest_below) * 100.0 / interest_points
            above_nhnm_interest_pct = np.sum(interest_above) * 100.0 / interest_points
            
            # Debug information to verify our calculation
            print(f"ROI: {min_period:.3f}s-{max_period:.3f}s ({min_freq:.1f}-{max_freq:.1f}Hz)")
            print(f"Total points in ROI: {interest_points}")
            print(f"Within models in ROI: {np.sum(interest_within)} points ({within_models_interest_pct:.1f}%)")
            print(f"Below NLNM in ROI: {np.sum(interest_below)} points ({below_nlnm_interest_pct:.1f}%)")
            print(f"Above NHNM in ROI: {np.sum(interest_above)} points ({above_nhnm_interest_pct:.1f}%)")
        else:
            within_models_interest_pct = 0.0
            below_nlnm_interest_pct = 0.0
            above_nhnm_interest_pct = 0.0
            print("No points found in the range of interest")
        
        # More robust visual check of ROI compliance
        # Count points by dividing the ROI into 10 equal logarithmic segments
        visual_compliance = 0.0
        if interest_points > 0:
            log_min_period = np.log10(min_period)
            log_max_period = np.log10(max_period)
            log_segments = np.logspace(log_min_period, log_max_period, 11)
            
            segment_compliance = []
            total_weighted_compliance = 0.0
            total_weight = 0.0
            
            for i in range(10):
                seg_min = log_segments[i]
                seg_max = log_segments[i+1]
                seg_mask = (period >= seg_min) & (period <= seg_max)
                
                seg_points = np.sum(seg_mask)
                if seg_points > 0:
                    seg_within_pct = np.sum(within_models & seg_mask) * 100.0 / seg_points
                    segment_compliance.append(seg_within_pct)
                    
                    # Weight by number of points in segment
                    total_weighted_compliance += seg_within_pct * seg_points
                    total_weight += seg_points
                    
                    print(f"Segment {i+1} ({seg_min:.3f}s-{seg_max:.3f}s): {seg_points} points, {seg_within_pct:.1f}% within models")
            
            # Calculate weighted average
            if total_weight > 0:
                visual_compliance = total_weighted_compliance / total_weight
                print(f"Visual segment compliance (weighted average): {visual_compliance:.1f}%")
                
            # If we have individual segment compliances, also calculate simple average
            if segment_compliance:
                simple_avg = sum(segment_compliance) / len(segment_compliance)
                print(f"Visual segment compliance (simple average): {simple_avg:.1f}%")
        
        # Identify problematic frequency bands - only consider issues in the range of interest
        issue_ranges = []
        
        # Function to find continuous ranges where condition is True
        def find_ranges(condition_mask):
            if not np.any(condition_mask):
                return []
                
            # Get indices where condition is true
            indices = np.where(condition_mask)[0]
            
            # Find breaks in consecutive indices
            breaks = np.where(np.diff(indices) > 1)[0]
            
            # Form ranges from the breaks
            ranges = []
            i_start = 0
            
            for i_end in breaks:
                start_idx = indices[i_start]
                end_idx = indices[i_end]
                ranges.append((period[start_idx], period[end_idx]))
                i_start = i_end + 1
                
            # Handle the last range
            if i_start < len(indices):
                start_idx = indices[i_start]
                end_idx = indices[-1]
                ranges.append((period[start_idx], period[end_idx]))
                
            return ranges
        
        # Find ranges below NLNM in ROI
        below_nlnm_ranges = find_ranges(interest_below)
        for start_period, end_period in below_nlnm_ranges:
            # Only report ranges that span at least 10% of a decade
            if end_period / start_period > 1.1:
                issue_ranges.append({
                    'type': 'below_nlnm',
                    'start_period': start_period,
                    'end_period': end_period,
                    'start_freq': 1.0/end_period,
                    'end_freq': 1.0/start_period,
                    'description': f"Below NLNM from {1.0/end_period:.2f} to {1.0/start_period:.2f} Hz"
                })
        
        # Find ranges above NHNM in ROI
        above_nhnm_ranges = find_ranges(interest_above)
        for start_period, end_period in above_nhnm_ranges:
            if end_period / start_period > 1.1:
                issue_ranges.append({
                    'type': 'above_nhnm',
                    'start_period': start_period,
                    'end_period': end_period,
                    'start_freq': 1.0/end_period,
                    'end_freq': 1.0/start_period,
                    'description': f"Above NHNM from {1.0/end_period:.2f} to {1.0/start_period:.2f} Hz"
                })
        
        # Calculate compliance score, prioritizing ROI
        # Use 80% weight for ROI, 20% for full spectrum
        interest_compliance = 10.0 - (above_nhnm_interest_pct * 0.15) - (below_nlnm_interest_pct * 0.05)
        full_compliance = 10.0 - (above_nhnm_pct * 0.1) - (below_nlnm_pct * 0.03)
        
        # If visual segment check shows high compliance, boost the score
        if visual_compliance > 90:
            interest_compliance = max(interest_compliance, 8.0)
        
        compliance_score = (interest_compliance * 0.8) + (full_compliance * 0.2)
        compliance_score = max(0.0, min(10.0, compliance_score))
        
        # Compile results
        results = {
            'within_models_pct': within_models_pct,  # Full spectrum
            'below_nlnm_pct': below_nlnm_pct,
            'above_nhnm_pct': above_nhnm_pct,
            'within_models_interest_pct': within_models_interest_pct,  # Range of interest
            'below_nlnm_interest_pct': below_nlnm_interest_pct,
            'above_nhnm_interest_pct': above_nhnm_interest_pct,
            'issue_ranges': issue_ranges,
            'compliance_score': compliance_score,
            'compliance_rating': 'Good' if compliance_score >= 7.0 else 
                                'Fair' if compliance_score >= 5.0 else 'Poor',
            'visual_compliance': visual_compliance
        }
        
        return results

    def detect_events(self, trace, sta_length=3.0, lta_length=30.0, threshold=3.5):
        """
        Detect events in a seismic trace using STA/LTA algorithm.
        
        Parameters:
        trace (obspy.Trace): Seismic trace
        sta_length (float): Short-term average window length in seconds
        lta_length (float): Long-term average window length in seconds
        threshold (float): Trigger threshold for STA/LTA ratio
        
        Returns:
        list: List of (start, end) sample indices for detected events
        """
        # Make a copy and process for trigger detection
        tr_filt = trace.copy()
        tr_filt.detrend('linear')
        tr_filt.taper(0.05)
        
        # For event detection, use a bandpass filter that emphasizes earthquake signals
        # Local earthquakes typically show energy in 1-20 Hz band
        try:
            tr_filt.filter('bandpass', freqmin=1.0, freqmax=20.0, corners=4)
        except ValueError as e:
            print(f"Filtering error: {e}. Using raw data.")
        
        # Calculate STA/LTA
        sampling_rate = tr_filt.stats.sampling_rate
        sta_samples = int(sta_length * sampling_rate)
        lta_samples = int(lta_length * sampling_rate)
        
        # Check if trace is long enough for analysis
        if len(tr_filt.data) < lta_samples * 2:
            print(f"Trace too short for event detection: {len(tr_filt.data)} samples")
            return []
        
        # Compute STA/LTA
        try:
            cft = classic_sta_lta(tr_filt.data, sta_samples, lta_samples)
            
            # Get triggers - threshold for activation, lower threshold for deactivation
            triggers = trigger_onset(cft, threshold, threshold * 0.5)
            
            # If we found triggers, add buffer around them
            if len(triggers) > 0:
                # Add buffer around triggers (10s before and after)
                buffer_samples = int(10 * sampling_rate)
                buffered_triggers = []
                
                for start, end in triggers:
                    buffered_start = max(0, start - buffer_samples)
                    buffered_end = min(len(tr_filt.data), end + buffer_samples)
                    buffered_triggers.append([buffered_start, buffered_end])
                
                # Merge overlapping triggers
                if buffered_triggers:
                    merged_triggers = [buffered_triggers[0]]
                    for current in buffered_triggers[1:]:
                        previous = merged_triggers[-1]
                        # If current trigger starts before previous ends, merge them
                        if current[0] <= previous[1]:
                            previous[1] = max(previous[1], current[1])
                        else:
                            merged_triggers.append(current)
                    
                    print(f"Detected {len(triggers)} events, merged to {len(merged_triggers)} segments")
                    return merged_triggers
                
            return triggers
        
        except Exception as e:
            print(f"Error in event detection: {e}")
            return []

    def load_data(self):
        # Check if miniseed file is selected
        mseed_path = self.miniseed_path.text()
        if not mseed_path:
            QMessageBox.warning(self, "Warning", "Please select a MiniSEED file.")
            return

        # Load miniseed file
        try:
            self.status_bar.showMessage(f"Loading {mseed_path}...")
            self.stream = read(mseed_path)
            self.status_bar.showMessage(f"Loaded {len(self.stream)} traces from {mseed_path}")

            # Load inventory if provided
            inv_path = self.inventory_path.text()
            if inv_path:
                try:
                    self.inventory = read_inventory(inv_path)
                    self.status_bar.showMessage(f"Loaded inventory from {inv_path}")
                except Exception as e:
                    self.status_bar.showMessage(f"Error loading inventory: {str(e)}")
                    self.inventory = None

            # Update station selector
            self.update_station_selector()

            # Switch to Analysis tab
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load MiniSEED file: {str(e)}")

    def fetch_data(self):
        # Get FDSN parameters
        server = self.fdsn_server.currentText()

        if server == "Custom":
            server_url = self.custom_server_url.text()
            if not server_url:
                QMessageBox.warning(self, "Warning", "Please enter a custom server URL.")
                return

        network = self.network.text()
        station = self.station.text()
        location = self.location.text()
        channel = self.channel.text()

        try:
            # Parse the datetime string with full format including hours, minutes, seconds
            start_time_str = self.start_time.text().strip()
            
            # Handle different possible formats
            try:
                # Try with full format first
                start_time = obspy.UTCDateTime(start_time_str)
            except Exception as e1:
                try:
                    # Try with just date if no time provided
                    if len(start_time_str.split()) == 1:
                        # If only date is given, add 00:00:00
                        start_time = obspy.UTCDateTime(f"{start_time_str} 00:00:00")
                    else:
                        # Re-raise the original error
                        raise e1
                except Exception as e2:
                    # Show detailed error message
                    QMessageBox.warning(
                        self, 
                        "Warning", 
                        f"Invalid datetime format: {str(e2)}\n\n"
                        f"Please use format: YYYY-MM-DD HH:MM:SS\n"
                        f"Example: 2025-03-06 08:30:00"
                    )
                    return
        except Exception as e:
            QMessageBox.warning(
                self, 
                "Warning", 
                f"Invalid datetime format: {str(e)}\n\n"
                f"Please use format: YYYY-MM-DD HH:MM:SS\n"
                f"Example: 2025-03-06 08:30:00"
            )
            return

        # Calculate end time based on duration
        end_time = start_time + self.duration.value() * 3600

        # Fetch data
        try:
            self.status_bar.showMessage(f"Connecting to {server} server...")

            if server == "Custom":
                client = Client(server_url)
            else:
                client = Client(server)

            self.status_bar.showMessage(f"Fetching waveform data...")
            self.stream = client.get_waveforms(
                network, station, location, channel,
                start_time, end_time
            )

            self.status_bar.showMessage(f"Fetching station metadata...")
            self.inventory = client.get_stations(
                network=network, station=station, location=location, channel=channel,
                starttime=start_time, endtime=end_time, level="response"
            )

            self.status_bar.showMessage(f"Fetched {len(self.stream)} traces from {server}")

            # Update station selector
            self.update_station_selector()
            
            # Update data overview
            self.update_data_overview()

            # Switch to Analysis tab
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch data: {str(e)}")
            self.status_bar.showMessage("Error fetching data")

    def update_station_selector(self):
        if not self.stream:
            return

        # Get unique stations
        stations = set()
        for tr in self.stream:
            stations.add(f"{tr.stats.network}.{tr.stats.station}")

        # Update the selector
        self.station_selector.clear()
        self.station_selector.addItems(sorted(stations))

    def run_analysis(self):
        if not self.stream:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load or fetch data first.")
            return
                
        # Get analysis parameters
        min_freq = self.min_freq.value()
        max_freq = self.max_freq.value()
        
        # Convert frequency to period
        min_period = 1.0 / max_freq
        max_period = 1.0 / min_freq
        
        # Get New Low and High Noise Models
        nlnm_periods, nlnm_power = get_nlnm()
        nhnm_periods, nhnm_power = get_nhnm()
        
        # Determine which preset is currently active based on the frequency values
        range_label = "Selected Frequency Range"
        if hasattr(self, 'current_preset_name') and self.current_preset_name != "Custom":
            preset_name = self.current_preset_name.split('(')[0].strip()
            range_label = f"{preset_name} Range"
        
        # Group traces by station
        stations = {}
        for tr in self.stream:
            station_id = f"{tr.stats.network}.{tr.stats.station}"
            if station_id not in stations:
                stations[station_id] = []
            stations[station_id].append(tr)
            
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(stations))
        self.progress_bar.setValue(0)
        
        # Results container
        results = []
        
        # Clear previous plots
        self.station_plots = {}  # Reset plot cache
        
        # Process each station
        output_dir = Path(self.output_dir.text())
        output_dir.mkdir(exist_ok=True)
        
        for i, (station_id, traces) in enumerate(stations.items()):
            self.status_bar.showMessage(f"Processing station {station_id}...")
            
            # Create a new figure for this station
            station_fig = Figure(figsize=(10, 8))
            ax = station_fig.add_subplot(111)
            
            # Process each component
            for tr in traces:
                try:
                    # Make a copy to avoid modifying original
                    trace = tr.copy()
                    
                    # Skip if trace is too short
                    if len(trace.data) < 10 * trace.stats.sampling_rate:
                        continue
                    
                    # Detrend and taper the data
                    trace.detrend('linear')
                    trace.taper(0.05)
                    
                    # Use event detection to remove earthquakes
                    # To this:
                    use_event_detection = self.use_event_detection.isChecked()  # Use value from UI

                    # And when calling detect_events:
                    triggers = self.detect_events(
                        trace,
                        sta_length=self.sta_length.value(),
                        lta_length=self.lta_length.value(),
                        threshold=self.trigger_threshold.value()
                    )

                    # Also modify the buffer creation:
                    sampling_rate = trace.stats.sampling_rate
                    buffer_samples = int(self.event_buffer.value() * sampling_rate)

                    if use_event_detection:
                        triggers = self.detect_events(trace)
                        if triggers:
                            self.status_bar.showMessage(f"Detected {len(triggers)} events in {tr.id}, removing them from analysis...")
                            quiet_mask = self.create_quiet_mask(trace, triggers)
                            
                            # Apply mask to remove event data
                            quiet_data = trace.data[quiet_mask]
                            
                            # Only use if we have enough data left
                            if len(quiet_data) > trace.stats.sampling_rate * self.segment_length.value():
                                quiet_trace = trace.copy()
                                quiet_trace.data = quiet_data
                                trace = quiet_trace
                            else:
                                print(f"Not enough data left after event removal in {tr.id}")
                    
                    # If we have inventory data, try to use it for response removal
                    if self.inventory:
                        try:
                            # Remove instrument response to get to acceleration
                            trace.remove_response(
                                inventory=self.inventory,
                                output="ACC",  # Convert to acceleration
                                water_level=60,  # dB water level for smoothing
                                pre_filt=(0.005, 0.006, 30.0, 35.0)  # Pre-filter to avoid edge effects
                            )
                        except Exception as e:
                            print(f"Could not remove response for {tr.id}: {e}")
                    else:
                        # If no response, assume the data is already in counts and needs conversion
                        # This is a simplified scaling that assumes a generic sensitivity
                        # For real analysis, you should use actual instrument responses
                        print(f"No inventory provided. Using simplified scaling for {tr.id}")
                        
                        # Scale by sampling rate to approximate differentiation (velocity to acceleration)
                        # This is a very rough approximation and should be replaced with proper response
                        trace.data = np.gradient(trace.data, 1.0/trace.stats.sampling_rate)
                        
                        # Apply a nominal sensitivity factor (this would be instrument-specific)
                        # For a typical seismometer, sensitivity might be around 1000 counts/m/s
                        nominal_sensitivity = 1000.0  # counts per m/s
                        trace.data = trace.data / nominal_sensitivity
                    
                    # Use robust PSD calculation
                    use_robust_psd = True  # Could be a checkbox in the UI
                    if use_robust_psd:
                        try:
                            f, power, conf_low, conf_high = self.calculate_robust_psd(
                                trace, 
                                self.segment_length.value(),
                                self.overlap.value(),
                                robust_method='median'
                            )
                            
                            # Convert to period
                            non_zero_f = f[f > 0]  # Avoid division by zero
                            period = 1.0 / non_zero_f
                            power = power[f > 0]
                            conf_low = conf_low[f > 0]
                            conf_high = conf_high[f > 0]
                            
                            # Sort by period
                            sort_idx = np.argsort(period)
                            period = period[sort_idx]
                            power = power[sort_idx]
                            conf_low = conf_low[sort_idx]
                            conf_high = conf_high[sort_idx]
                            
                            # Plot with confidence intervals
                            line = ax.semilogx(period, power, label=f"{tr.stats.channel}")[0]
                            color = line.get_color()
                            ax.fill_between(period, conf_low[sort_idx], conf_high[sort_idx], color=color, alpha=0.2)
                        except Exception as e:
                            print(f"Error in robust PSD calculation: {e}")
                            use_robust_psd = False
                    
                    if not use_robust_psd:
                        # Fall back to original PSD calculation if robust method fails
                        nperseg = min(int(trace.stats.sampling_rate * self.segment_length.value()), 
                                    len(trace.data)//4)
                        if nperseg < 10:
                            continue
                        
                        # Calculate in (m/s^2)^2/Hz
                        f, Pxx = signal.welch(
                            trace.data, 
                            fs=trace.stats.sampling_rate, 
                            nperseg=nperseg, 
                            noverlap=int(nperseg * self.overlap.value()),
                            scaling='density',  # Power spectral density scaling
                            detrend='linear'
                        )
                        
                        # Convert PSD to dB
                        power = 10.0 * np.log10(Pxx)
                        
                        # Convert frequency to period
                        non_zero_f = f[f > 0]  # Avoid division by zero
                        period = 1.0 / non_zero_f
                        power = power[f > 0]
                        
                        # Sort by period for proper plotting
                        sort_idx = np.argsort(period)
                        period = period[sort_idx]
                        power = power[sort_idx]
                        
                        # Plot the PSD
                        ax.semilogx(period, power, label=f"{tr.stats.channel}")
                    
                    # After calculating and plotting the PSD, but before local seismicity metrics
                    # Check compliance with noise models
                    noise_compliance = self.check_noise_model_compliance(
                        period, power, 
                        nlnm_periods, nlnm_power, 
                        nhnm_periods, nhnm_power
                    )

                    # Add annotations to the plot for compliance issues
                    for issue in noise_compliance['issue_ranges']:
                        if issue['type'] == 'above_nhnm':
                            ax.axvspan(issue['start_period'], issue['end_period'], 
                                    alpha=0.2, color='red', hatch='////')
                            midpoint = np.sqrt(issue['start_period'] * issue['end_period'])
                            ax.text(midpoint, -60, "Above NHNM", 
                                color='red', ha='center', fontsize=8, rotation=90)
                        elif issue['type'] == 'below_nlnm':
                            ax.axvspan(issue['start_period'], issue['end_period'], 
                                    alpha=0.2, color='blue', hatch='\\\\\\\\')
                            midpoint = np.sqrt(issue['start_period'] * issue['end_period'])
                            ax.text(midpoint, -180, "Below NLNM", 
                                color='blue', ha='center', fontsize=8, rotation=90)
                    
                    # Calculate noise metrics for local seismicity range
                    local_mask = (period >= min_period) & (period <= max_period)
                    
                    if np.any(local_mask):
                        local_period = period[local_mask]
                        local_power = power[local_mask]
                        
                        # Interpolate NLNM and NHNM to match our periods
                        nlnm_interp = np.interp(local_period, nlnm_periods, nlnm_power)
                        nhnm_interp = np.interp(local_period, nlnm_periods, nhnm_power)
                        
                        # Calculate metrics
                        mean_power = np.mean(local_power)
                        max_power = np.max(local_power)
                        min_power = np.min(local_power)
                        
                        # Distance from NLNM in dB
                        snr = np.mean(local_power - nlnm_interp)
                        
                        # Store the result
                        result = {
                            'station': station_id,
                            'channel': tr.stats.channel,
                            'mean_power_db': mean_power,
                            'min_power_db': min_power,
                            'max_power_db': max_power,
                            'snr_to_nlnm': snr,
                            'suitable_for_local': False,  # Will be set later
                            'start_time': tr.stats.starttime.datetime,
                            'end_time': tr.stats.endtime.datetime,
                            'duration_hours': (tr.stats.endtime - tr.stats.starttime) / 3600
                        }
                        
                        # Add compliance info to result dictionary
                        result.update({
                            'within_models_pct': noise_compliance['within_models_pct'],
                            'below_nlnm_pct': noise_compliance['below_nlnm_pct'],
                            'above_nhnm_pct': noise_compliance['above_nhnm_pct'],
                            'within_models_interest_pct': noise_compliance['within_models_interest_pct'],  # Add the new metric
                            'compliance_score': noise_compliance['compliance_score'],
                            'compliance_rating': noise_compliance['compliance_rating'],
                            'has_compliance_issues': len(noise_compliance['issue_ranges']) > 0,
                            'visual_compliance': noise_compliance['visual_compliance']
                        })
                        
                        # Apply frequency weighting if enabled
                        if self.use_weighting.isChecked():
                            peak_freq = self.peak_weight_freq.value()
                            # Convert period to frequency for weighting
                            f_values = 1.0 / local_period
                            # Apply gaussian-like weighting centered on peak_freq
                            weights = np.exp(-0.5 * ((f_values - peak_freq) / (peak_freq / 3)) ** 2)
                            # Normalize weights
                            weights = weights / np.sum(weights)
                            # Apply weights to calculate weighted SNR
                            weighted_power = np.sum(local_power * weights)
                            weighted_nlnm = np.sum(nlnm_interp * weights)
                            result['weighted_snr'] = weighted_power - weighted_nlnm
                            result['snr_to_nlnm'] = result['weighted_snr']  # Use weighted SNR
                        
                        results.append(result)
                        
                except Exception as e:
                    print(f"Error processing {tr.id}: {e}")
                    import traceback
                    traceback.print_exc()
                    
            # Add noise models to plot
            ax.semilogx(nlnm_periods, nlnm_power, 'k--', label='NLNM')
            ax.semilogx(nhnm_periods, nhnm_power, 'k-', label='NHNM')
            
            # Add plot details
            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.set_xlabel('Period (s)')
            ax.set_ylabel('Power (dB rel. to 1 (m/s²)²/Hz)')
            ax.set_title(f'Noise PSD for Station {station_id}')
            ax.legend()
            
            # Set y-axis limits to a reasonable range for seismic data
            ax.set_ylim([-200, -50])
            ax.set_xlim([0.01, 1000])
            
            # Add selected frequency range with appropriate label
            ax.axvspan(min_period, max_period, alpha=0.2, color='red')
            ax.text(np.sqrt(min_period * max_period), -195, 
                    f"Range: {min_freq}-{max_freq} Hz", color='red')

            
            # Store the figure for later use
            self.station_plots[station_id] = station_fig
            
            # Save the plot if requested
            if self.save_plots.isChecked():
                station_fig.savefig(output_dir / f"{station_id}_noise_psd.png")
                
            # Update progress
            self.progress_bar.setValue(i + 1)


        
        # Determine station suitability based on selected method
        if results:
            # Extract all SNR values for analysis
            all_snr_values = [result['snr_to_nlnm'] for result in results]
            
            if self.relative_ranking.isChecked():
                # Relative ranking approach - select top N% of stations
                percentile = self.percentile_threshold.value()
                threshold = np.percentile(all_snr_values, percentile)
                for result in results:
                    result['suitable_for_local'] = result['snr_to_nlnm'] <= threshold
                self.status_bar.showMessage(f"Using relative threshold: {threshold:.2f} dB (top {percentile}%)")
                
            elif self.statistical_method.isChecked():
                # Statistical approach - select stations within X std dev of the best
                min_snr = min(all_snr_values)
                std_dev = np.std(all_snr_values)
                threshold = min_snr + (self.std_dev_threshold.value() * std_dev)
                for result in results:
                    result['suitable_for_local'] = result['snr_to_nlnm'] <= threshold
                self.status_bar.showMessage(
                    f"Using statistical threshold: {threshold:.2f} dB "
                    f"({self.std_dev_threshold.value()} std dev from best: {min_snr:.2f} dB)"
                )
                
            else:
                # Absolute threshold approach
                threshold = self.noise_threshold.value()
                for result in results:
                    result['suitable_for_local'] = result['snr_to_nlnm'] <= threshold
                self.status_bar.showMessage(f"Using absolute threshold: {threshold:.2f} dB")
            
            # Clean up temporary fields used for analysis
            for result in results:
                if 'local_power' in result:
                    del result['local_power']
                if 'local_period' in result:
                    del result['local_period']
                if 'nlnm_interp' in result:
                    del result['nlnm_interp']
                if 'weighted_snr' in result and result['weighted_snr'] != result['snr_to_nlnm']:
                    # If we used weighting, keep both values for the report
                    pass
                else:
                    if 'weighted_snr' in result:
                        del result['weighted_snr']
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Update results table
        self.results_df = pd.DataFrame(results) if results else pd.DataFrame()
        self.update_results_table()
        
        # Update plot if there are results
        if not self.results_df.empty and self.station_selector.currentText():
            self.update_plot(self.station_selector.currentText())
            
        # Switch to Results tab
        self.tabs.setCurrentIndex(2)
        
        # Show summary
        if not self.results_df.empty:
            suitable_stations = len(self.results_df[self.results_df['suitable_for_local']]['station'].unique())
            total_stations = len(self.results_df['station'].unique())
            self.status_bar.showMessage(
                f"Analysis complete. {suitable_stations} of {total_stations} stations "
                f"suitable for local seismicity using {threshold:.1f} dB threshold."
            )
        else:
            self.status_bar.showMessage("Analysis complete. No valid results.")

    def calculate_robust_psd(self, trace, segment_length, overlap, nperseg=None, robust_method='median'):
        """
        Calculate power spectral density using robust methods.
        
        Parameters:
        trace (obspy.Trace): Seismic trace
        segment_length (float): Segment length in seconds
        overlap (float): Overlap between segments (0-1)
        nperseg (int, optional): Number of points per segment
        robust_method (str): 'median' or 'mean' to use as aggregation method
        
        Returns:
        tuple: (frequencies, psd, confidence_low, confidence_high)
        """
        if nperseg is None:
            nperseg = int(trace.stats.sampling_rate * segment_length)
        
        # Ensure nperseg is valid
        nperseg = min(nperseg, len(trace.data)//2)
        if nperseg < 10:
            raise ValueError("Segment length too short for PSD calculation")
        
        noverlap = int(nperseg * overlap)
        
        # Calculate spectra for each segment
        frequencies, times, Sxx = signal.spectrogram(
            trace.data,
            fs=trace.stats.sampling_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )
        
        # Transpose to get [segments, frequencies]
        Sxx = Sxx.T
        
        # Calculate robust statistics
        if robust_method == 'median':
            psd = np.median(Sxx, axis=0)
            # Estimate confidence intervals (16th and 84th percentiles ≈ 1 sigma)
            conf_low = np.percentile(Sxx, 16, axis=0)
            conf_high = np.percentile(Sxx, 84, axis=0)
        else:  # default to mean
            psd = np.mean(Sxx, axis=0)
            # Standard deviation for confidence
            std_dev = np.std(Sxx, axis=0)
            conf_low = psd - std_dev
            conf_high = psd + std_dev
        
        # Convert to dB
        psd_db = 10 * np.log10(psd)
        conf_low_db = 10 * np.log10(conf_low)
        conf_high_db = 10 * np.log10(conf_high)
        
        return frequencies, psd_db, conf_low_db, conf_high_db

    def update_results_table(self):
        # Clear the table
        self.results_table.setRowCount(0)

        if self.results_df is None or self.results_df.empty:
            return

        # Add a column for compliance rating
        self.results_table.setColumnCount(7)  # Increase from 6 to 7
        self.results_table.setHorizontalHeaderLabels([
            "Station", "Channel", "Mean Power (dB)", "SNR to NLNM (dB)", 
            "Suitable", "Duration (hours)", "Compliance"  # Added "Compliance"
        ])

        # Fill the table with results
        self.results_table.setRowCount(len(self.results_df))

        for i, (_, row) in enumerate(self.results_df.iterrows()):
            self.results_table.setItem(i, 0, QTableWidgetItem(row['station']))
            self.results_table.setItem(i, 1, QTableWidgetItem(row['channel']))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{row['mean_power_db']:.2f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{row['snr_to_nlnm']:.2f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem("Yes" if row['suitable_for_local'] else "No"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{row['duration_hours']:.2f}"))
            
            # Add compliance rating
            if 'compliance_rating' in row:
                compliance_item = QTableWidgetItem(row['compliance_rating'])
                self.results_table.setItem(i, 6, compliance_item)
                
                # Color based on rating
                if row['compliance_rating'] == 'Good':
                    compliance_item.setBackground(Qt.green)
                elif row['compliance_rating'] == 'Fair':
                    compliance_item.setBackground(Qt.yellow)
                else:  # Poor
                    compliance_item.setBackground(Qt.red)
            else:
                self.results_table.setItem(i, 6, QTableWidgetItem("N/A"))

            # Set background color based on suitability for rows 0-5
            for j in range(6):
                if row['suitable_for_local']:
                    self.results_table.item(i, j).setBackground(Qt.green)
                else:
                    self.results_table.item(i, j).setBackground(Qt.red)

        # Adjust column widths
        self.results_table.resizeColumnsToContents()

    def update_plot(self, station_id):
        if not station_id or self.results_df is None or self.results_df.empty:
            return

        # Clear the figure
        self.figure.clear()

        # If we have a cached plot for this station, use it
        if station_id in self.station_plots:
            # Copy the content from the cached figure to our display figure
            cached_fig = self.station_plots[station_id]
            for ax in cached_fig.get_axes():
                new_ax = self.figure.add_subplot(111)

                # Copy lines and their properties
                for line in ax.get_lines():
                    new_ax.plot(line.get_xdata(), line.get_ydata(),
                               color=line.get_color(),
                               linestyle=line.get_linestyle(),
                               label=line.get_label())

                # Copy axis properties
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.set_title(ax.get_title())
                new_ax.set_xscale(ax.get_xscale())
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
                new_ax.grid(True, which="both", ls="-", alpha=0.5)

                # Copy patches (like spans)
                for patch in ax.patches:
                    if patch.__class__.__name__ == 'Polygon':  # For span
                        new_ax.axvspan(patch.get_xy()[0][0], patch.get_xy()[1][0],
                                      alpha=0.2, color='red')

                # Add legend
                new_ax.legend()

                # We only need to copy the first subplot
                break
        else:
            # If no cached plot, create a new one with minimal information
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"No plot available for {station_id}",
                   ha='center', va='center', transform=ax.transAxes)

        # Draw the canvas
        self.canvas.draw()

    # Then, replace the entire export_report method with this implementation:

    def export_report(self):
        if self.results_df is None or self.results_df.empty:
            QMessageBox.warning(self, "Warning", "No analysis results available. Please run analysis first.")
            return

        output_dir = Path(self.output_dir.text())
        output_dir.mkdir(exist_ok=True)

        # Save CSV if requested
        if self.save_csv.isChecked():
            csv_path = output_dir / "noise_results.csv"
            self.results_df.to_csv(csv_path, index=False)
            self.status_bar.showMessage(f"Saved results to {csv_path}")

        # Save HTML report if requested
        if self.save_html.isChecked():
            html_path = output_dir / "noise_report.html"

            # Check if we have the visual_compliance column
            has_visual_compliance = 'visual_compliance' in self.results_df.columns
            
            # Calculate statistics
            total_stations = len(self.results_df['station'].unique())
            suitable_stations = len(self.results_df[self.results_df['suitable_for_local']]['station'].unique())
            
            # Check if we have compliance data
            has_compliance_data = 'compliance_rating' in self.results_df.columns
            
            # Create a summary of compliance ratings if available
            compliance_summary = ""
            if has_compliance_data:
                # Count unique stations for each compliance rating
                good_stations = set()
                fair_stations = set()
                poor_stations = set()
                
                for _, row in self.results_df.iterrows():
                    if row['compliance_rating'] == 'Good':
                        good_stations.add(row['station'])
                    elif row['compliance_rating'] == 'Fair':
                        fair_stations.add(row['station'])
                    elif row['compliance_rating'] == 'Poor':
                        poor_stations.add(row['station'])
                
                good_count = len(good_stations)
                fair_count = len(fair_stations)
                poor_count = len(poor_stations)
                
                compliance_summary = f"""
                    <div class="summary-section">
                        <h3>Noise Model Compliance</h3>
                        <div class="compliance-summary">
                            <div class="compliance-item good">
                                <span class="count">{good_count}</span>
                                <span class="label">Good</span>
                            </div>
                            <div class="compliance-item fair">
                                <span class="count">{fair_count}</span>
                                <span class="label">Fair</span>
                            </div>
                            <div class="compliance-item poor">
                                <span class="count">{poor_count}</span>
                                <span class="label">Poor</span>
                            </div>
                        </div>
                    </div>
                """

            # Generate HTML report with enhanced CSS and JS for interactivity
            html_report = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Station Noise Analysis Report</title>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        margin: 20px; 
                        color: #333;
                        line-height: 1.6;
                    }}
                    h1, h2, h3 {{ 
                        color: #2c3e50; 
                        margin-top: 20px;
                    }}
                    .container {{
                        max-width: 1200px;
                        margin: 0 auto;
                    }}
                    .summary {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                        margin-bottom: 30px;
                        background-color: #f9f9f9;
                        padding: 20px;
                        border-radius: 5px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }}
                    .summary-section {{
                        flex: 1;
                        min-width: 250px;
                    }}
                    .station-summary {{
                        display: flex;
                        align-items: center;
                        gap: 15px;
                    }}
                    .station-count {{
                        font-size: 2.5em;
                        font-weight: bold;
                        color: #3498db;
                    }}
                    .suitable-count {{
                        font-size: 2.5em;
                        font-weight: bold;
                        color: #2ecc71;
                    }}
                    .parameters {{
                        background-color: #f5f5f5;
                        padding: 15px;
                        border-radius: 5px;
                    }}
                    .parameters ul {{
                        margin: 0;
                        padding-left: 20px;
                    }}
                    .controls {{
                        background-color: #f5f5f5;
                        padding: 15px 20px;
                        border-radius: 5px;
                        margin-bottom: 20px;
                        display: flex;
                        flex-wrap: wrap;
                        gap: 15px;
                        align-items: center;
                    }}
                    .search-box {{
                        flex-grow: 1;
                        min-width: 200px;
                    }}
                    .search-box input {{
                        width: 100%;
                        padding: 8px 12px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        font-size: 16px;
                    }}
                    .filter-controls {{
                        display: flex;
                        gap: 10px;
                        flex-wrap: wrap;
                    }}
                    .filter-button {{
                        padding: 8px 12px;
                        background-color: #fff;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        cursor: pointer;
                        transition: all 0.2s;
                    }}
                    .filter-button:hover {{
                        background-color: #f0f0f0;
                    }}
                    .filter-button.active {{
                        background-color: #3498db;
                        color: white;
                        border-color: #3498db;
                    }}
                    .sort-controls {{
                        display: flex;
                        gap: 10px;
                    }}
                    .sort-select {{
                        padding: 8px 12px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }}
                    table {{ 
                        border-collapse: collapse; 
                        width: 100%; 
                        margin: 20px 0;
                    }}
                    th, td {{ 
                        padding: 12px 8px; 
                        text-align: left; 
                        border-bottom: 1px solid #ddd; 
                    }}
                    th {{ 
                        background-color: #3498db; 
                        color: white;
                        cursor: pointer;
                        position: relative;
                    }}
                    th:hover {{
                        background-color: #2980b9;
                    }}
                    th::after {{
                        content: "";
                        position: absolute;
                        right: 8px;
                        top: 50%;
                        transform: translateY(-50%);
                    }}
                    th.sort-asc::after {{
                        content: "▲";
                    }}
                    th.sort-desc::after {{
                        content: "▼";
                    }}
                    tr:nth-child(even) {{ 
                        background-color: #f2f2f2; 
                    }}
                    tr:hover {{
                        background-color: #e3f2fd;
                    }}
                    tr.hidden {{
                        display: none;
                    }}
                    .suitable {{ 
                        background-color: #d4edda !important; 
                    }}
                    .unsuitable {{ 
                        background-color: #f8d7da !important; 
                    }}
                    .plot-section {{
                        margin-top: 40px;
                    }}
                    .station-plots {{
                        display: flex;
                        flex-wrap: wrap;
                        gap: 20px;
                    }}
                    .station-plot {{
                        margin: 20px 0;
                        padding: 15px;
                        background-color: #f9f9f9;
                        border-radius: 5px;
                        width: 100%;
                        transition: all 0.3s ease;
                    }}
                    .plot-grid-view .station-plot {{
                        width: calc(50% - 20px);
                    }}
                    .station-plot img {{
                        max-width: 100%;
                        height: auto;
                        display: block;
                        margin: 0 auto;
                        border: 1px solid #ddd;
                        cursor: pointer;
                        transition: transform 0.3s ease;
                    }}
                    .station-plot img:hover {{
                        transform: scale(1.02);
                    }}
                    .compliance-summary {{
                        display: flex;
                        gap: 15px;
                        margin-top: 10px;
                    }}
                    .compliance-item {{
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                        flex: 1;
                    }}
                    .compliance-item .count {{
                        display: block;
                        font-size: 1.8em;
                        font-weight: bold;
                    }}
                    .good {{ background-color: #d4edda; color: #155724; }}
                    .fair {{ background-color: #fff3cd; color: #856404; }}
                    .poor {{ background-color: #f8d7da; color: #721c24; }}
                    .compliance-info {{
                        margin-left: 10px;
                        color: #666;
                        font-style: italic;
                    }}
                    .compliance-metrics {{
                        display: flex;
                        flex-direction: column;
                        gap: 4px;
                    }}
                    .metric-row {{
                        display: flex;
                        align-items: center;
                    }}
                    .metric-label {{
                        min-width: 150px;
                        color: #555;
                    }}
                    .metric-value {{
                        font-weight: bold;
                        margin-right: 5px;
                    }}
                    .metric-explanation {{
                        font-size: 0.85em;
                        color: #666;
                        font-style: italic;
                    }}
                    .visual-metric {{
                        background-color: #e8f4f8;
                        border-radius: 3px;
                        padding: 3px 6px;
                    }}
                    .view-controls {{
                        display: flex;
                        gap: 10px;
                        margin-bottom: 15px;
                    }}
                    .view-button {{
                        padding: 8px 12px;
                        background-color: #fff;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        gap: 5px;
                    }}
                    .view-button.active {{
                        background-color: #3498db;
                        color: white;
                    }}
                    .view-button svg {{
                        width: 16px;
                        height: 16px;
                    }}
                    .modal {{
                        display: none;
                        position: fixed;
                        z-index: 1000;
                        left: 0;
                        top: 0;
                        width: 100%;
                        height: 100%;
                        background-color: rgba(0,0,0,0.7);
                        overflow: auto;
                    }}
                    .modal-content {{
                        position: relative;
                        margin: auto;
                        padding: 0;
                        width: 90%;
                        max-width: 1200px;
                        top: 50%;
                        transform: translateY(-50%);
                    }}
                    .modal-image {{
                        display: block;
                        width: 100%;
                        max-height: 90vh;
                        object-fit: contain;
                    }}
                    .close {{
                        position: absolute;
                        top: 10px;
                        right: 25px;
                        color: white;
                        font-size: 35px;
                        font-weight: bold;
                        cursor: pointer;
                    }}
                    .explanation-section {{
                        background-color: #f8f9fa;
                        padding: 20px;
                        border-radius: 5px;
                        margin-top: 30px;
                    }}
                    .back-to-top {{
                        position: fixed;
                        bottom: 20px;
                        right: 20px;
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 50%;
                        width: 50px;
                        height: 50px;
                        font-size: 20px;
                        cursor: pointer;
                        display: none;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                        z-index: 100;
                    }}
                    .back-to-top:hover {{
                        background-color: #2980b9;
                    }}
                    /* Responsive adjustments */
                    @media (max-width: 768px) {{
                        .plot-grid-view .station-plot {{
                            width: 100%;
                        }}
                        .controls {{
                            flex-direction: column;
                            align-items: stretch;
                        }}
                        .filter-controls, .sort-controls {{
                            flex-direction: column;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Station Noise Analysis Report</h1>
                    <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="summary">
                        <div class="summary-section">
                            <h3>Station Summary</h3>
                            <div class="station-summary">
                                <div>
                                    <span class="station-count">{total_stations}</span>
                                    <div>Total stations</div>
                                </div>
                                <div>
                                    <span class="suitable-count">{suitable_stations}</span>
                                    <div>Suitable for local seismicity</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="summary-section">
                            <h3>Analysis Parameters</h3>
                            <div class="parameters">
                                <ul>
                                    <li>Frequency range: {self.min_freq.value()} - {self.max_freq.value()} Hz</li>
                                    <li>SNR threshold: {self.noise_threshold.value()} dB</li>
                                    <li>Segment length: {self.segment_length.value()} s</li>
                                    <li>Overlap: {self.overlap.value()}</li>
                                    <li>Analysis method: {
                                        "Absolute Threshold" if self.absolute_threshold.isChecked() else
                                        "Relative Ranking" if self.relative_ranking.isChecked() else
                                        "Statistical Method"
                                    }</li>
                                </ul>
                            </div>
                        </div>
                        
                        {compliance_summary if has_compliance_data else ""}
                    </div>
                    
                    <h2>Detailed Results</h2>
                    
                    <div class="controls">
                        <div class="search-box">
                            <input type="text" id="searchInput" placeholder="Search for stations, channels...">
                        </div>
                        <div class="filter-controls">
                            <button class="filter-button active" data-filter="all">All</button>
                            <button class="filter-button" data-filter="suitable">Suitable only</button>
                            <button class="filter-button" data-filter="unsuitable">Unsuitable only</button>
                            {'''
                            <button class="filter-button" data-filter="good">Good compliance</button>
                            <button class="filter-button" data-filter="fair">Fair compliance</button>
                            <button class="filter-button" data-filter="poor">Poor compliance</button>
                            ''' if has_compliance_data else ""}
                        </div>
                        <div class="sort-controls">
                            <select id="sortSelect" class="sort-select">
                                <option value="station-asc">Station (A-Z)</option>
                                <option value="station-desc">Station (Z-A)</option>
                                <option value="snr-asc">SNR (Low to High)</option>
                                <option value="snr-desc">SNR (High to Low)</option>
                                <option value="power-asc">Power (Low to High)</option>
                                <option value="power-desc">Power (High to Low)</option>
                                {'''
                                <option value="compliance-asc">Compliance Score (Low to High)</option>
                                <option value="compliance-desc">Compliance Score (High to Low)</option>
                                ''' if has_compliance_data else ""}
                            </select>
                        </div>
                    </div>
                    
                    <table id="resultsTable">
                        <thead>
                            <tr>
                                <th data-sort="station">Station</th>
                                <th data-sort="channel">Channel</th>
                                <th data-sort="power" class="sort-none">Mean Power (dB)</th>
                                <th data-sort="snr" class="sort-none">SNR to NLNM (dB)</th>
                                <th data-sort="suitable" class="sort-none">Suitable</th>
                                <th data-sort="duration" class="sort-none">Duration (hours)</th>
                                {f'''
                                <th data-sort="compliance" class="sort-none">Compliance</th>
                                <th>Model Compliance Metrics</th>
                                ''' if has_compliance_data else ""}
                            </tr>
                        </thead>
                        <tbody>
            """

            # Add rows for each result
            for _, row in self.results_df.iterrows():
                row_class = "suitable" if row['suitable_for_local'] else "unsuitable"
                
                compliance_cols = ""
                compliance_data_attr = ""
                if has_compliance_data:
                    compliance_class = row['compliance_rating'].lower()
                    compliance_data_attr = f' data-compliance="{compliance_class}"'
                    
                    # Create a more structured display of compliance metrics
                    segmented_analysis_row = ""
                    if has_visual_compliance:
                        segmented_analysis_row = f"""
                        <div class="metric-row">
                            <span class="metric-label">Segmented analysis:</span>
                            <span class="metric-value visual-metric">{row['visual_compliance']:.1f}%</span>
                            <span class="metric-explanation">(Weighted by bands)</span>
                        </div>
                        """
                    
                    compliance_metrics = f"""
                    <div class="compliance-metrics">
                        <div class="metric-row">
                            <span class="metric-label">Full spectrum:</span>
                            <span class="metric-value">{row['within_models_pct']:.1f}%</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-label">Range of interest:</span>
                            <span class="metric-value">{row['within_models_interest_pct']:.1f}%</span>
                        </div>
                        {segmented_analysis_row}
                    </div>
                    """
                    
                    compliance_cols = f'''
                    <td class="{compliance_class}" data-value="{row['compliance_score']}">{row["compliance_rating"]}</td>
                    <td>{compliance_metrics}</td>
                    '''
                    
                html_report += f"""
                <tr class="{row_class}"{compliance_data_attr} data-station="{row['station']}">
                    <td>{row['station']}</td>
                    <td>{row['channel']}</td>
                    <td data-value="{row['mean_power_db']}">{row['mean_power_db']:.2f}</td>
                    <td data-value="{row['snr_to_nlnm']}">{row['snr_to_nlnm']:.2f}</td>
                    <td data-value="{1 if row['suitable_for_local'] else 0}">{"Yes" if row['suitable_for_local'] else "No"}</td>
                    <td data-value="{row['duration_hours']}">{row['duration_hours']:.2f}</td>
                    {compliance_cols}
                </tr>
                """

            # Continue with the rest of the HTML
            html_report += """
                        </tbody>
                    </table>
                    
                    <div class="plot-section">
                        <h2>PSD Plots</h2>
                        <p>Power spectral density plots showing noise levels compared to Peterson's New Low and High Noise Models.
                        Red shaded areas indicate the target frequency range for local seismicity.</p>
                        
                        <div class="view-controls">
                            <button id="listViewBtn" class="view-button active">
                                <svg viewBox="0 0 24 24">
                                    <path fill="currentColor" d="M3,4H21V8H3V4M3,10H21V14H3V10M3,16H21V20H3V16Z" />
                                </svg>
                                List View
                            </button>
                            <button id="gridViewBtn" class="view-button">
                                <svg viewBox="0 0 24 24">
                                    <path fill="currentColor" d="M3,11H11V3H3M3,21H11V13H3M13,21H21V13H13M13,3V11H21V3" />
                                </svg>
                                Grid View
                            </button>
                        </div>
                        
                        <div id="stationPlots" class="station-plots">
            """

            # Add images for each station - EMBED IMAGES HERE
            for station in self.results_df['station'].unique():
                # Get suitability info for this station
                station_rows = self.results_df[self.results_df['station'] == station]
                suitable = any(station_rows['suitable_for_local'])
                
                # Get compliance info if available
                compliance_info = ""
                compliance_data_attr = ""
                if has_compliance_data:
                    ratings = station_rows['compliance_rating'].unique()
                    compliance_class = ratings[0].lower() if len(ratings) == 1 else ""
                    compliance_info = f"<span class='compliance-info'>Compliance: {', '.join(ratings)}</span>"
                    compliance_data_attr = f' data-compliance="{",".join(r.lower() for r in ratings)}"'
                
                # Check if we have a cached plot for this station
                if station in self.station_plots:
                    # Convert the figure to a base64 string
                    figure = self.station_plots[station]
                    buffer = BytesIO()
                    figure.savefig(buffer, format='png', dpi=100)
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.read()).decode('utf-8')
                    img_data_url = f"data:image/png;base64,{img_str}"
                    
                    html_report += f"""
                    <div class="station-plot{' suitable' if suitable else ' unsuitable'}" 
                        data-station="{station}"{compliance_data_attr}>
                        <h3>Station {station} {"(Suitable)" if suitable else "(Not Suitable)"} {compliance_info}</h3>
                        <img src="{img_data_url}" alt="PSD Plot for {station}" onclick="openModal(this.src)">
                    </div>
                    """
                else:
                    # If we don't have a cached plot but we saved the file
                    if self.save_plots.isChecked():
                        # Check if the file exists
                        plot_file = Path(self.output_dir.text()) / f"{station}_noise_psd.png"
                        if plot_file.exists():
                            # Read the file and convert to base64
                            with open(plot_file, 'rb') as f:
                                img_str = base64.b64encode(f.read()).decode('utf-8')
                            img_data_url = f"data:image/png;base64,{img_str}"
                            
                            html_report += f"""
                            <div class="station-plot{' suitable' if suitable else ' unsuitable'}" 
                                data-station="{station}"{compliance_data_attr}>
                                <h3>Station {station} {"(Suitable)" if suitable else "(Not Suitable)"} {compliance_info}</h3>
                                <img src="{img_data_url}" alt="PSD Plot for {station}" onclick="openModal(this.src)">
                            </div>
                            """
                        else:
                            # No plot available
                            html_report += f"""
                            <div class="station-plot{' suitable' if suitable else ' unsuitable'}" 
                                data-station="{station}"{compliance_data_attr}>
                                <h3>Station {station} {"(Suitable)" if suitable else "(Not Suitable)"} {compliance_info}</h3>
                                <p>No PSD plot available for this station.</p>
                            </div>
                            """
                    else:
                        # No plot available
                        html_report += f"""
                        <div class="station-plot{' suitable' if suitable else ' unsuitable'}" 
                            data-station="{station}"{compliance_data_attr}>
                            <h3>Station {station} {"(Suitable)" if suitable else "(Not Suitable)"} {compliance_info}</h3>
                            <p>No PSD plot available for this station.</p>
                        </div>
                        """

            # Add explanatory section based on what data is available
            explanation_section = """
                    <div class="explanation-section">
                        <h2>Compliance Metrics Explanation</h2>
                        <p>The compliance metrics show how well a station's noise profile stays within the standard 
                        noise model boundaries (NLNM and NHNM):</p>
                        <ul>
                            <li><strong>Full spectrum:</strong> Percentage of data points across the entire frequency spectrum that fall within the noise models.</li>
                            <li><strong>Range of interest:</strong> Percentage of data points within your selected frequency range of interest that fall within the noise models.</li>
            """
            
            if has_visual_compliance:
                explanation_section += """
                            <li><strong>Segmented analysis:</strong> A more robust metric that divides your frequency range into logarithmic segments 
                            and calculates compliance for each segment, weighted by the number of data points in each segment. This helps ensure 
                            that compliance across the entire frequency range is considered, even if data points are not evenly distributed.</li>
                """
            
            explanation_section += """
                        </ul>
                    </div>
            """
            
            html_report += explanation_section

            html_report += """
                </div>
                
                <!-- Modal for image zoom -->
                <div id="imageModal" class="modal">
                    <span class="close" onclick="closeModal()">&times;</span>
                    <div class="modal-content">
                        <img id="modalImage" class="modal-image">
                    </div>
                </div>
                
                <!-- Back to top button -->
                <button id="backToTop" class="back-to-top" title="Back to top">↑</button>
                
                <script>
                    // Search functionality
                    const searchInput = document.getElementById('searchInput');
                    const resultsTable = document.getElementById('resultsTable');
                    const tableRows = resultsTable.querySelectorAll('tbody tr');
                    const stationPlots = document.querySelectorAll('.station-plot');
                    const filterButtons = document.querySelectorAll('.filter-button');
                    const sortSelect = document.getElementById('sortSelect');
                    const thElements = document.querySelectorAll('th[data-sort]');
                    const listViewBtn = document.getElementById('listViewBtn');
                    const gridViewBtn = document.getElementById('gridViewBtn');
                    const stationPlotsContainer = document.getElementById('stationPlots');
                    const backToTopBtn = document.getElementById('backToTop');
                    
                    // Current filter state
                    let currentFilter = 'all';
                    
                    // Function to filter table rows and station plots
                    function filterElements() {
                        const searchTerm = searchInput.value.toLowerCase();
                        
                        // Filter table rows
                        tableRows.forEach(row => {
                            const station = row.querySelector('td:first-child').textContent.toLowerCase();
                            const channel = row.querySelector('td:nth-child(2)').textContent.toLowerCase();
                            const isSuitable = row.classList.contains('suitable');
                            
                            // Get compliance if available
                            const complianceAttr = row.getAttribute('data-compliance');
                            
                            let show = (station.includes(searchTerm) || channel.includes(searchTerm));
                            
                            // Apply current filter
                            if (currentFilter === 'suitable' && !isSuitable) show = false;
                            if (currentFilter === 'unsuitable' && isSuitable) show = false;
                            if (currentFilter === 'good' && complianceAttr !== 'good') show = false;
                            if (currentFilter === 'fair' && complianceAttr !== 'fair') show = false;
                            if (currentFilter === 'poor' && complianceAttr !== 'poor') show = false;
                            
                            row.classList.toggle('hidden', !show);
                        });
                        
                        // Filter station plots
                        stationPlots.forEach(plot => {
                            const station = plot.getAttribute('data-station').toLowerCase();
                            const isSuitable = plot.classList.contains('suitable');
                            
                            // Get compliance if available
                            const complianceAttr = plot.getAttribute('data-compliance');
                            
                            let show = station.includes(searchTerm);
                            
                            // Apply current filter
                            if (currentFilter === 'suitable' && !isSuitable) show = false;
                            if (currentFilter === 'unsuitable' && isSuitable) show = false;
                            if (currentFilter === 'good' && !complianceAttr?.includes('good')) show = false;
                            if (currentFilter === 'fair' && !complianceAttr?.includes('fair')) show = false;
                            if (currentFilter === 'poor' && !complianceAttr?.includes('poor')) show = false;
                            
                            plot.style.display = show ? '' : 'none';
                        });
                    }
                    
                    // Function to sort table rows
                    function sortTable(sortBy, direction) {
                        const tbody = resultsTable.querySelector('tbody');
                        const rows = Array.from(tbody.querySelectorAll('tr'));
                        
                        rows.sort((a, b) => {
                            let aValue, bValue;
                            
                            if (sortBy === 'station') {
                                aValue = a.cells[0].textContent;
                                bValue = b.cells[0].textContent;
                            } else if (sortBy === 'channel') {
                                aValue = a.cells[1].textContent;
                                bValue = b.cells[1].textContent;
                            } else {
                                // For numeric columns, use the data-value attribute
                                const index = sortBy === 'power' ? 2 : 
                                            sortBy === 'snr' ? 3 : 
                                            sortBy === 'suitable' ? 4 : 
                                            sortBy === 'duration' ? 5 : 
                                            sortBy === 'compliance' ? 6 : 0;
                                
                                aValue = parseFloat(a.cells[index].getAttribute('data-value') || 0);
                                bValue = parseFloat(b.cells[index].getAttribute('data-value') || 0);
                            }
                            
                            // Compare based on direction
                            if (direction === 'asc') {
                                return aValue > bValue ? 1 : -1;
                            } else {
                                return aValue < bValue ? 1 : -1;
                            }
                        });
                        
                        // Update DOM
                        rows.forEach(row => tbody.appendChild(row));
                        
                        // Update sort indicators
                        thElements.forEach(th => {
                            const thisSortBy = th.getAttribute('data-sort');
                            th.classList.remove('sort-asc', 'sort-desc', 'sort-none');
                            if (thisSortBy === sortBy) {
                                th.classList.add(direction === 'asc' ? 'sort-asc' : 'sort-desc');
                            } else {
                                th.classList.add('sort-none');
                            }
                        });
                    }
                    
                    // Event Listeners
                    searchInput.addEventListener('input', filterElements);
                    
                    filterButtons.forEach(button => {
                        button.addEventListener('click', () => {
                            filterButtons.forEach(btn => btn.classList.remove('active'));
                            button.classList.add('active');
                            currentFilter = button.getAttribute('data-filter');
                            filterElements();
                        });
                    });
                    
                    sortSelect.addEventListener('change', () => {
                        const [sortBy, direction] = sortSelect.value.split('-');
                        sortTable(sortBy, direction);
                    });
                    
                    thElements.forEach(th => {
                        th.addEventListener('click', () => {
                            const sortBy = th.getAttribute('data-sort');
                            let direction = 'asc';
                            
                            if (th.classList.contains('sort-asc')) {
                                direction = 'desc';
                            } else if (th.classList.contains('sort-desc')) {
                                direction = 'asc';
                            }
                            
                            sortTable(sortBy, direction);
                            
                            // Update the sort select value to match the current sort
                            sortSelect.value = `${sortBy}-${direction}`;
                        });
                    });
                    
                    // View switching
                    listViewBtn.addEventListener('click', () => {
                        gridViewBtn.classList.remove('active');
                        listViewBtn.classList.add('active');
                        stationPlotsContainer.classList.remove('plot-grid-view');
                    });
                    
                    gridViewBtn.addEventListener('click', () => {
                        listViewBtn.classList.remove('active');
                        gridViewBtn.classList.add('active');
                        stationPlotsContainer.classList.add('plot-grid-view');
                    });
                    
                    // Modal image viewer
                    function openModal(imgSrc) {
                        const modal = document.getElementById('imageModal');
                        const modalImg = document.getElementById('modalImage');
                        modal.style.display = 'block';
                        modalImg.src = imgSrc;
                    }
                    
                    function closeModal() {
                        document.getElementById('imageModal').style.display = 'none';
                    }
                    
                    // Close modal when clicking outside the image
                    window.onclick = function(event) {
                        const modal = document.getElementById('imageModal');
                        if (event.target === modal) {
                            closeModal();
                        }
                    };
                    
                    // Back to top button
                    window.onscroll = function() {scrollFunction()};
                    
                    function scrollFunction() {
                        if (document.body.scrollTop > 300 || document.documentElement.scrollTop > 300) {
                            backToTopBtn.style.display = "block";
                        } else {
                            backToTopBtn.style.display = "none";
                        }
                    }
                    
                    backToTopBtn.addEventListener('click', () => {
                        document.body.scrollTop = 0; // For Safari
                        document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
                    });
                    
                    // Initialize: sort by default and apply filters
                    sortTable('station', 'asc');
                    filterElements();
                </script>
            </body>
            </html>
            """

            # Save HTML report
            with open(html_path, "w") as f:
                f.write(html_report)

            self.status_bar.showMessage(f"Saved interactive HTML report to {html_path}")

        QMessageBox.information(self, "Success", f"Report exported to {output_dir}")


class AnalysisWorker(QThread):
    """Worker thread for running analysis in the background"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, stream, params):
        super().__init__()
        self.stream = stream
        self.params = params

    def run(self):
        try:
            # Get analysis parameters
            min_freq = self.params['min_freq']
            max_freq = self.params['max_freq']
            noise_threshold = self.params['noise_threshold']
            segment_length = self.params['segment_length']
            overlap = self.params['overlap']
            save_plots = self.params['save_plots']
            output_dir = self.params['output_dir']

            # Convert frequency to period
            min_period = 1.0 / max_freq
            max_period = 1.0 / min_freq

            # Get New Low and High Noise Models
            nlnm_periods, nlnm_power = get_nlnm()
            nhnm_periods, nhnm_power = get_nhnm()

            # Group traces by station
            stations = {}
            for tr in self.stream:
                station_id = f"{tr.stats.network}.{tr.stats.station}"
                if station_id not in stations:
                    stations[station_id] = []
                stations[station_id].append(tr)

            # Results container
            results = []

            # Process each station
            for i, (station_id, traces) in enumerate(stations.items()):
                # Emit progress
                self.progress.emit(i)

                # Create a figure for this station
                fig = Figure(figsize=(10, 8))
                ax = fig.add_subplot(111)

                # Process each component
                for tr in traces:
                    try:
                        # Make a copy to avoid modifying original
                        trace = tr.copy()

                        # Skip if trace is too short
                        if len(trace.data) < 10 * trace.stats.sampling_rate:
                            continue

                        # Detrend and taper the data
                        trace.detrend('linear')
                        trace.taper(0.05)

                        # Calculate PSD using scipy's welch method
                        nperseg = min(int(trace.stats.sampling_rate * segment_length),
                                     len(trace.data)//4)
                        if nperseg < 10:
                            continue

                        f, Pxx = signal.welch(
                            trace.data,
                            fs=trace.stats.sampling_rate,
                            nperseg=nperseg,
                            noverlap=int(nperseg * overlap),
                            scaling='density'
                        )

                        # Convert to dB
                        Pxx_db = 10 * np.log10(Pxx)

                        # Convert frequency to period for comparison with noise models
                        period = 1.0 / f[1:]  # Skip the DC component (f=0)
                        power = Pxx_db[1:]

                        # Plot the PSD
                        ax.semilogx(period, power, label=f"{tr.stats.channel}")

                        # Calculate noise metrics
                        local_mask = (period >= min_period) & (period <= max_period)

                        if np.any(local_mask):
                            local_power = power[local_mask]
                            local_period = period[local_mask]

                            # Calculate metrics
                            mean_power = np.mean(local_power)
                            max_power = np.max(local_power)
                            min_power = np.min(local_power)

                            # Compare with noise models
                            nlnm_interp = np.interp(local_period, nlnm_periods, nlnm_power)
                            nhnm_interp = np.interp(local_period, nhnm_periods, nhnm_power)

                            # Calculate distance from New Low Noise Model
                            snr = np.mean(local_power - nlnm_interp)

                            # Determine if suitable for local seismicity
                            is_suitable = snr < noise_threshold

                            results.append({
                                'station': station_id,
                                'channel': tr.stats.channel,
                                'mean_power_db': mean_power,
                                'min_power_db': min_power,
                                'max_power_db': max_power,
                                'snr_to_nlnm': snr,
                                'suitable_for_local': is_suitable,
                                'start_time': tr.stats.starttime.datetime,
                                'end_time': tr.stats.endtime.datetime,
                                'duration_hours': (tr.stats.endtime - tr.stats.starttime) / 3600
                            })

                    except Exception as e:
                        print(f"Error processing {tr.id}: {e}")

                # Add noise models to plot
                ax.semilogx(nlnm_periods, nlnm_power, 'k--', label='NLNM')
                ax.semilogx(nhnm_periods, nhnm_power, 'k-', label='NHNM')

                # Add plot details
                ax.grid(True, which="both", ls="-", alpha=0.5)
                ax.set_xlabel('Period (s)')
                ax.set_ylabel('Power (dB rel. to 1 (m/s²)²/Hz)')
                ax.set_title(f'Noise PSD for Station {station_id}')
                ax.legend()

                # Set y-axis limits to a reasonable range for seismic data
                ax.set_ylim([-200, -50])

                # Add this line to limit the x-axis to a maximum of 1000 seconds:
                ax.set_xlim([0.01, 1000])  # Limit period range from 0.01s to 1000s

                # The rest of the code remains unchanged:
                # Add selected frequency range with appropriate label
                ax.axvspan(min_period, max_period, alpha=0.2, color='red')
                ax.text(np.sqrt(min_period * max_period), -195, 
                        f"Range: {min_freq}-{max_freq} Hz", color='red')


                # Save the plot if requested
                if save_plots:
                    output_path = Path(output_dir) / f"{station_id}_noise_psd.png"
                    fig.savefig(output_path)

            # Convert results to DataFrame
            results_df = pd.DataFrame(results) if results else pd.DataFrame()

            # Emit finished signal with results
            self.finished.emit(results_df)

        except Exception as e:
            self.error.emit(str(e))


def main():
    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = StationNoiseAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
