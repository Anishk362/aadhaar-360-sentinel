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
      title: 'Aadhaar Darpan Intel V3.0',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF00695C), // Govt Teal
          brightness: Brightness.light,
        ),
        textTheme: GoogleFonts.latoTextTheme(),
        scaffoldBackgroundColor: const Color(0xFFF5F7FA), // Light Grey BG
      ),
      home: const DashboardScreen(),
    );
  }
}

// --- MAIN DASHBOARD SCREEN ---
class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  // --- 1. STATE VARIABLES ---
  Map<String, dynamic>? metadata; // Stores State -> [District list]
  Map<String, dynamic>? auditData; // Stores the final Intelligence Report
  
  String? selectedState;
  String? selectedDistrict;
  bool isLoading = false;

  // --- 2. API CONFIGURATION ---
  // USE THIS FOR ANDROID EMULATOR:
  final String baseUrl = "http://127.0.0.1:5001/api";
  
  // USE THIS FOR WEB (CHROME):
  // final String baseUrl = "http://192.168.39.123:5001/api"; 

  @override
  void initState() {
    super.initState();
    _fetchMetadata(); // Load dropdowns immediately
  }

  // --- 3. FETCH METADATA (States & Districts) ---
  Future<void> _fetchMetadata() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/metadata'));
      if (response.statusCode == 200) {
        setState(() {
          metadata = json.decode(response.body)['metadata'];
        });
      }
    } catch (e) {
      print("CRITICAL ERROR: Is Python running? $e");
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Backend Offline. Run 'python app.py'")),
      );
    }
  }

  // --- 4. FETCH INTELLIGENCE REPORT ---
  Future<void> _fetchAuditReport() async {
    if (selectedState == null) return;

    setState(() => isLoading = true);
    
    // If district is null, send empty string to get State Aggregate
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

  // --- 5. UI BUILDER ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Aadhaar Darpan: Intel V3.0", 
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        backgroundColor: const Color(0xFF00695C),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: Colors.white),
            onPressed: () {
              selectedState = null;
              selectedDistrict = null;
              auditData = null;
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

  // --- WIDGET: DROPDOWNS ---
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
            const Text("Deep Dive Analysis", style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            const SizedBox(height: 15),
            
            // STATE DROPDOWN
            Container(
              decoration: BoxDecoration(color: Colors.grey.shade100, borderRadius: BorderRadius.circular(8)),
              child: DropdownButtonFormField<String>(
                decoration: const InputDecoration(labelText: "Select State", border: OutlineInputBorder()),
                value: selectedState,
                items: metadata?.keys.map((state) {
                  return DropdownMenuItem(value: state, child: Text(state));
                }).toList(),
                onChanged: (value) {
                  setState(() {
                    selectedState = value;
                    selectedDistrict = null; // Reset district
                    _fetchAuditReport(); // Auto-fetch state view
                  });
                },
              ),
            ),
            const SizedBox(height: 15),

            // DISTRICT DROPDOWN
            Container(
              decoration: BoxDecoration(color: Colors.grey.shade100, borderRadius: BorderRadius.circular(8)),
              child: DropdownButtonFormField<String>(
                decoration: const InputDecoration(labelText: "Select District", border: OutlineInputBorder()),
                value: selectedDistrict,
                // Logic: List is empty if no state is selected
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
            ),
          ],
        ),
      ),
    );
  }

  // --- WIDGET: DASHBOARD ---
Widget _buildReportView() {
    final location = auditData!['location'] ?? "Unknown";
    final cards = auditData!['cards'];
    
    // Parsing Security Data
    final secStatus = cards['security']['status']; 
    final secMsg = "Volume: ${cards['security']['mobile_update_volume']}"; 
    
    // Logic: Convert raw ratio to percentage for UI visualization
    final double femaleRatio = (cards['inclusivity']['female_enrolment_pct'] as num).toDouble();
    final double femalePercentage = femaleRatio * 100;
    final double malePercentage = 100 - femalePercentage;
    
    // Parsing Inclusivity Data - Target benchmark is 50%
    final incMsg = "Female: ${femalePercentage.toStringAsFixed(1)}% (${(50 - femalePercentage).toStringAsFixed(1)}% gap)";

    // Parsing Forecast Data from Member 3's ML Model
    final forecast = List<dynamic>.from(cards['efficiency']['biometric_traffic_trend']);

    // UI Feedback Colors based on Risk Logic
    final isCritical = secStatus == "CRITICAL";
    final securityColor = isCritical ? Colors.red.shade50 : Colors.green.shade50;
    final securityIconColor = isCritical ? Colors.red : Colors.green;

    return Column(
      children: [
        Text("$location Analysis", style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
        const SizedBox(height: 10),
        const Text("Gender Inclusivity Ratio", style: TextStyle(fontSize: 12, color: Colors.grey)),
        
        // Data-Driven PieChart: Visualizing the Gender Gap
        SizedBox(
          height: 180,
          child: PieChart(
            PieChartData(
              sections: [
                PieChartSectionData(
                  value: femalePercentage,
                  color: Colors.purple,
                  title: "F ${femalePercentage.toStringAsFixed(1)}%",
                  radius: 55,
                  titleStyle: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                PieChartSectionData(
                  value: malePercentage,
                  color: const Color(0xFF00695C), // Govt Teal Theme
                  title: "M ${malePercentage.toStringAsFixed(1)}%",
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

        // Strategic Audit Alert Cards
        Row(
          children: [
            Expanded(
              child: _buildStatusCard("Security Status", secMsg, securityColor, securityIconColor),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: _buildStatusCard("Inclusivity", incMsg, Colors.blue.shade50, Colors.blue),
            ),
          ],
        ),

        const SizedBox(height: 20),
        const Text("Efficiency Trend (3 Months Forecast)", style: TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 10),

        // ML-Driven Bar Chart: Predicted Biometric Traffic
        Container(
          height: 200,
          padding: const EdgeInsets.all(16),
          child: BarChart(
            BarChartData(
              alignment: BarChartAlignment.spaceAround,
              maxY: (forecast.last as num).toDouble() * 1.3, // Dynamic scaling
              barTouchData: BarTouchData(enabled: true),
              titlesData: const FlTitlesData(show: false),
              borderData: FlBorderData(show: false),
              gridData: const FlGridData(show: false),
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

  // --- HELPER FUNCTIONS ---
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

  Widget _buildStatusCard(String title, String msg, Color bgColor, Color textColor) {
    return Container(
      padding: const EdgeInsets.all(12),
      height: 120,
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: textColor.withOpacity(0.3)),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(title, style: TextStyle(color: textColor, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text(msg, textAlign: TextAlign.center, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600)),
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