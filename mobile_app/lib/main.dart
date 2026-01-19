import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:fl_chart/fl_chart.dart';
import 'package:google_fonts/google_fonts.dart';

// --- APP ENTRY POINT ---
void main() {
  runApp(const AadhaarDarpanApp());
}

class AadhaarDarpanApp extends StatelessWidget {
  const AadhaarDarpanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Aadhaar Darpan Intel V4.0',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF00695C), // Govt Teal
          brightness: Brightness.light,
        ),
        textTheme: GoogleFonts.latoTextTheme(),
        scaffoldBackgroundColor: const Color(0xFFF5F7FA),
      ),
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  Map<String, dynamic>? metadata;
  Map<String, dynamic>? auditData;
  String? selectedState;
  String? selectedDistrict;
  bool isLoading = false;

  // --- API CONFIGURATION ---
  // Change this to your Computer's IP if testing on a real mobile device
  final String baseUrl = "http://127.0.0.1:5001/api";

  @override
  void initState() {
    super.initState();
    _fetchMetadata();
  }

  Future<void> _fetchMetadata() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/metadata'));
      if (response.statusCode == 200) {
        setState(() {
          metadata = json.decode(response.body)['metadata'];
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Backend Offline. Ensure Flask is running.")),
      );
    }
  }

  Future<void> _fetchAuditReport() async {
    if (selectedState == null) return;
    setState(() => isLoading = true);
    String districtParam = selectedDistrict ?? "";
    try {
      final uri = Uri.parse('$baseUrl/audit?state=$selectedState&district=$districtParam');
      final response = await http.get(uri);
      if (response.statusCode == 200) {
        setState(() {
          auditData = json.decode(response.body);
        });
      }
    } catch (e) {
      print("Audit Fetch Error: $e");
    } finally {
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Aadhaar Darpan: Intel V4.0", 
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF00695C),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: () {
              setState(() {
                selectedState = null;
                selectedDistrict = null;
                auditData = null;
              });
              _fetchMetadata();
            },
          )
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            _buildDeepDiveSection(),
            const SizedBox(height: 20),
            if (isLoading) 
              const Center(child: CircularProgressIndicator())
            else if (auditData != null)
              _buildReportView()
            else
              _buildEmptyState(),
          ],
        ),
      ),
    );
  }

  Widget _buildDeepDiveSection() {
    return Card(
      elevation: 2,
      color: Colors.white,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Regional Readiness Audit", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 15),
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: "Select State", border: OutlineInputBorder()),
              value: selectedState,
              items: metadata?.keys.map((state) => DropdownMenuItem(value: state, child: Text(state))).toList(),
              onChanged: (value) {
                setState(() {
                  selectedState = value;
                  selectedDistrict = null;
                  _fetchAuditReport();
                });
              },
            ),
            const SizedBox(height: 15),
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: "Select District", border: OutlineInputBorder()),
              value: selectedDistrict,
              items: selectedState == null 
                  ? [] 
                  : (metadata![selectedState] as List).map<DropdownMenuItem<String>>((district) {
                      return DropdownMenuItem(value: district.toString(), child: Text(district.toString()));
                    }).toList(),
              onChanged: (value) {
                setState(() {
                  selectedDistrict = value;
                  _fetchAuditReport();
                });
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildReportView() {
    final location = auditData!['location'] ?? "Unknown";
    final cards = auditData!['cards'];
    
    // --- PILLAR 2: SERVICE ACCESS RISK ---
    final secStatus = cards['security']['status']; 
    final double accessVal = (cards['security']['value'] as num).toDouble();
    final secMsg = "Active Access: $accessVal%"; 

    // --- PILLAR 1: CHILD SATURATION ---
    final double saturationRatio = (cards['inclusivity']['value'] as num).toDouble();
    final double youthPercentage = saturationRatio * 100;
    final double adultPercentage = 100 - youthPercentage;
    final String satStatus = cards['inclusivity']['status'];
    final incMsg = "Youth Onboarding: ${youthPercentage.toStringAsFixed(1)}%\nStatus: $satStatus";

    // --- PILLAR 3: INFRASTRUCTURE FORECAST ---
    final forecast = List<dynamic>.from(cards['efficiency']['biometric_traffic_trend']);

    final isCritical = secStatus == "CRITICAL" || satStatus == "CRITICAL";
    final statusColor = isCritical ? Colors.red.shade50 : Colors.green.shade50;
    final iconColor = isCritical ? Colors.red : Colors.green;

    return Column(
      children: [
        Text("$location Analysis", style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
        const SizedBox(height: 10),
        const Text("Generation Saturation Index", style: TextStyle(fontSize: 12, color: Colors.grey)),
        
        SizedBox(
          height: 220,
          child: PieChart(
            PieChartData(
              sections: [
                PieChartSectionData(
                  value: youthPercentage < 1 ? 1 : youthPercentage, // Force visibility
                  color: Colors.purple,
                  title: "Youth ${youthPercentage.toStringAsFixed(1)}%",
                  radius: 55,
                  titleStyle: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                PieChartSectionData(
                  value: adultPercentage,
                  color: const Color(0xFF00695C),
                  title: "Adults ${adultPercentage.toStringAsFixed(1)}%",
                  radius: 50,
                  titleStyle: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white),
                ),
              ],
              centerSpaceRadius: 40,
              sectionsSpace: 2,
            ),
          ),
        ),
        const SizedBox(height: 20),

        Row(
          children: [
            Expanded(
              child: _buildStatusCard("Service Access Risk", secMsg, statusColor, iconColor, Icons.wifi_tethering),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _buildStatusCard("Child Saturation", incMsg, Colors.blue.shade50, Colors.blue, Icons.child_care),
            ),
          ],
        ),

        const SizedBox(height: 20),
        const Text("Infrastructure Load Forecast (3 Mo)", style: TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 10),

        Container(
          height: 200,
          padding: const EdgeInsets.all(16),
          child: BarChart(
            BarChartData(
              alignment: BarChartAlignment.spaceAround,
              maxY: (forecast.last as num).toDouble() * 1.3,
              barTouchData: BarTouchData(enabled: true),
              titlesData: const FlTitlesData(show: false),
              borderData: FlBorderData(show: false),
              barGroups: [
                _buildBar(0, forecast[0], Colors.lightBlue.shade200),
                _buildBar(1, forecast[1], Colors.lightBlue.shade300),
                _buildBar(2, forecast[2], Colors.lightBlue.shade400),
              ],
            ),
          ),
        ),
      ],
    );
  }

  BarChartGroupData _buildBar(int x, dynamic y, Color color) {
    double value = (y as num).toDouble();
    return BarChartGroupData(x: x, barRods: [
      BarChartRodData(
        toY: value, 
        color: color, 
        width: 35, 
        borderRadius: BorderRadius.circular(6),
        backDrawRodData: BackgroundBarChartRodData(
          show: true, 
          toY: value * 1.3, 
          color: Colors.grey.shade100
        ),
      )
    ]);
  }

  Widget _buildStatusCard(String title, String msg, Color bgColor, Color textColor, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(12),
      height: 130,
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: textColor.withOpacity(0.3)),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: textColor, size: 24),
          const SizedBox(height: 4),
          Text(title, style: TextStyle(color: textColor, fontWeight: FontWeight.bold, fontSize: 13)),
          const SizedBox(height: 4),
          Text(msg, textAlign: TextAlign.center, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Container(
      padding: const EdgeInsets.all(40),
      child: Column(
        children: [
          Icon(Icons.analytics_outlined, size: 80, color: Colors.grey.shade300),
          const SizedBox(height: 10),
          const Text("Select a state to begin strategic audit.", style: TextStyle(color: Colors.grey)),
        ],
      ),
    );
  }
}