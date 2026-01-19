import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:fl_chart/fl_chart.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:countries_world_map/countries_world_map.dart'; // Added for India Map

void main() {
  runApp(const AadhaarDarpanApp());
}

class AadhaarDarpanApp extends StatelessWidget {
  const AadhaarDarpanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Aadhaar Darpan Intel V4.9',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF00695C),
          brightness: Brightness.light,
        ),
        textTheme: GoogleFonts.latoTextTheme(),
        scaffoldBackgroundColor: const Color(0xFFF0F4F4),
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
  Map<String, dynamic>? heatmapData;
  String? selectedState;
  String? selectedDistrict;
  bool isLoading = false;

  final String baseUrl = "http://127.0.0.1:5001/api";

  @override
  void initState() {
    super.initState();
    _fetchMetadata();
    _fetchHeatmap();
  }

  Future<void> _fetchMetadata() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/metadata')).timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        setState(() {
          metadata = json.decode(response.body)['metadata'];
        });
      }
    } catch (e) {
      _showSnackbar("Engine Offline: Ensure Sanitization API is running.");
    }
  }

  Future<void> _fetchHeatmap() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/heatmap'));
      if (response.statusCode == 200) {
        setState(() {
          heatmapData = json.decode(response.body)['data'];
        });
      }
    } catch (e) {
      debugPrint("Heatmap Error: $e");
    }
  }

  Future<void> _fetchAuditReport() async {
    if (selectedState == null) return;
    setState(() => isLoading = true);
    try {
      final uri = Uri.parse('$baseUrl/audit?state=$selectedState&district=${selectedDistrict ?? ""}');
      final response = await http.get(uri);
      if (response.statusCode == 200) {
        setState(() => auditData = json.decode(response.body));
      }
    } catch (e) {
      _showSnackbar("Data Desync: Audit Engine failed to process region.");
    } finally {
      setState(() => isLoading = false);
    }
  }

  void _showSnackbar(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg),
      backgroundColor: Colors.redAccent,
      behavior: SnackBarBehavior.floating,
    ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Column(
          children: [
            const Text("AADHAAR DARPAN", style: TextStyle(letterSpacing: 1.2, fontWeight: FontWeight.w900, color: Colors.white)),
            Text(selectedState == null ? "National Audit Hub" : "Regional Intelligence: $selectedState",
                style: const TextStyle(fontSize: 10, color: Colors.white70)),
          ],
        ),
        backgroundColor: const Color(0xFF00695C),
        centerTitle: true,
        elevation: 0,
        actions: [
          IconButton(icon: const Icon(Icons.refresh_rounded, color: Colors.white), onPressed: () {
            _fetchMetadata();
            _fetchHeatmap();
          }),
        ],
      ),
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF00695C), Color(0xFFF0F4F4)],
            stops: [0.1, 0.3],
          ),
        ),
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 16.0),
          child: Column(
            children: [
              const SizedBox(height: 10),
              _buildSelectionCard(),
              const SizedBox(height: 24),
              if (isLoading)
                const Center(child: Padding(padding: EdgeInsets.all(50), child: CircularProgressIndicator()))
              else if (auditData != null)
                _buildReportView()
              else
                _buildHomeScreen(),
              const SizedBox(height: 40),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSelectionCard() {
    return Card(
      elevation: 12,
      shadowColor: Colors.black26,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            DropdownButtonFormField<String>(
              decoration: InputDecoration(
                labelText: "Official 36 States/UTs",
                prefixIcon: const Icon(Icons.map_sharp, color: Color(0xFF00695C)),
                filled: true,
                fillColor: Colors.grey.shade50,
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
              ),
              value: selectedState,
              items: metadata?.keys.map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
              onChanged: (val) {
                setState(() {
                  selectedState = val;
                  selectedDistrict = null;
                  auditData = null;
                  _fetchAuditReport();
                });
              },
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              decoration: InputDecoration(
                labelText: "District Drilldown (Optional)",
                prefixIcon: const Icon(Icons.location_on, color: Colors.orange),
                filled: true,
                fillColor: Colors.grey.shade50,
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide.none),
              ),
              value: selectedDistrict,
              items: selectedState == null ? [] : (metadata![selectedState] as List).map((d) => DropdownMenuItem(value: d.toString(), child: Text(d.toString()))).toList(),
              onChanged: (val) {
                setState(() {
                  selectedDistrict = val;
                  _fetchAuditReport();
                });
              },
            ),
          ],
        ),
      ),
    );
  }

  // REPLACED GRID WITH INDIA MAP IDEA
  Widget _buildHomeScreen() {
    if (heatmapData == null) {
      return Column(
        children: [
          const SizedBox(height: 40),
          const CircularProgressIndicator(),
          const SizedBox(height: 20),
          const Text("Syncing National Pulse...", style: TextStyle(color: Colors.grey)),
        ],
      );
    }

    return Column(
      children: [
        const Text("NATIONAL HEALTH PULSE", style: TextStyle(fontSize: 16, fontWeight: FontWeight.w900, letterSpacing: 2)),
        const SizedBox(height: 20),
        
        // INTERACTIVE COLORED INDIA MAP
        
        SizedBox(
          height: 400,
          child: SimpleMap(
            instructions: SMapIndia.instructions,
            defaultColor: Colors.grey.shade300,
            colors: _generateMapColors(),
            callback: (id, name, taparea) {
              setState(() {
                selectedState = name.toUpperCase();
                selectedDistrict = null;
                _fetchAuditReport();
              });
            },
          ),
        ),
        
        const SizedBox(height: 25),
        _buildLegend(),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 30.0, vertical: 20),
          child: Text(
            "Choropleth visualization of all 36 entities. Tap any state for a regional deep dive.",
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.grey, fontSize: 11),
          ),
        ),
      ],
    );
  }

  // HELPER FOR MAP COLORS
  Map<String, Color> _generateMapColors() {
    Map<String, Color> mapColors = {};
    heatmapData?.forEach((state, details) {
      String status = details['status'];
      mapColors[state] = status == "SAFE" 
          ? Colors.green.withOpacity(0.8) 
          : (status == "WARNING" ? Colors.orange : Colors.red);
    });
    return mapColors;
  }

  Widget _buildLegend() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _legendItem("Critical", Colors.red),
        const SizedBox(width: 20),
        _legendItem("Warning", Colors.orange),
        const SizedBox(width: 20),
        _legendItem("Safe", Colors.green),
      ],
    );
  }

  Widget _legendItem(String label, Color color) {
    return Row(
      children: [
        Container(width: 10, height: 10, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
        const SizedBox(width: 6),
        Text(label, style: const TextStyle(fontSize: 10, color: Colors.grey, fontWeight: FontWeight.bold)),
      ],
    );
  }

  Widget _buildReportView() {
    final location = auditData!['location'] ?? "Report";
    final cards = auditData!['cards'];
    
    final double saturationRatio = (cards['inclusivity']['value'] as num).toDouble();
    final double youthPercentage = saturationRatio * 100;
    final double adultPercentage = 100 - youthPercentage;
    final forecast = List<dynamic>.from(cards['efficiency']['biometric_traffic_trend']);

    return Column(
      children: [
        Container(
          padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(30)),
          child: Text("AUDIT TARGET: $location", style: const TextStyle(fontWeight: FontWeight.w900, color: Color(0xFF00695C))),
        ),
        const SizedBox(height: 24),
        
        _buildSectionHeader("Pillar I: Generation Saturation", Icons.child_friendly),
        const SizedBox(height: 10),
        SizedBox(
          height: 200,
          child: PieChart(
            PieChartData(
              sections: [
                PieChartSectionData(
                  value: youthPercentage.clamp(0.1, 99.9), 
                  color: Colors.deepPurpleAccent,
                  title: "Youth: ${youthPercentage.toStringAsFixed(1)}%",
                  radius: 70,
                  titleStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                PieChartSectionData(
                  value: adultPercentage.clamp(0.1, 99.9),
                  color: const Color(0xFF00695C),
                  title: "Adults",
                  radius: 60,
                  titleStyle: const TextStyle(fontSize: 10, color: Colors.white70),
                ),
              ],
              centerSpaceRadius: 40,
            ),
          ),
        ),
        
        const SizedBox(height: 24),
        _buildSectionHeader("Pillar II: Service Access & Risk", Icons.security),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(child: _statusTile("Service Access", "${cards['security']['value']}%", cards['security']['status'])),
            const SizedBox(width: 12),
            Expanded(child: _statusTile("Inclusivity", cards['inclusivity']['status'], cards['inclusivity']['status'])),
          ],
        ),

        const SizedBox(height: 32),
        _buildSectionHeader("Pillar III: Predictive Load (90 Days)", Icons.trending_up),
        const SizedBox(height: 16),

        Container(
          height: 220,
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(20)),
          child: BarChart(
            BarChartData(
              alignment: BarChartAlignment.spaceAround,
              maxY: (forecast.last as num).toDouble() * 1.5,
              barGroups: List.generate(3, (i) => BarChartGroupData(x: i, barRods: [
                BarChartRodData(toY: (forecast[i] as num).toDouble(), color: Colors.orangeAccent, width: 35, borderRadius: const BorderRadius.vertical(top: Radius.circular(6)))
              ])),
              titlesData: const FlTitlesData(show: false),
              gridData: const FlGridData(show: false),
              borderData: FlBorderData(show: false),
            ),
          ),
        ),
        const SizedBox(height: 20),
        ElevatedButton.icon(
          onPressed: () => setState(() => auditData = null),
          icon: const Icon(Icons.arrow_back),
          label: const Text("Back to National Map"),
          style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF00695C), foregroundColor: Colors.white),
        )
      ],
    );
  }

  Widget _buildSectionHeader(String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, size: 18, color: Colors.grey),
        const SizedBox(width: 8),
        Text(title.toUpperCase(), style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold, color: Colors.grey, letterSpacing: 1.1)),
      ],
    );
  }

  Widget _statusTile(String title, String val, String status) {
    Color col = status == "SAFE" ? Colors.green : (status == "WARNING" ? Colors.orange : Colors.red);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border(left: BorderSide(color: col, width: 5)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: TextStyle(fontSize: 11, color: Colors.grey.shade600)),
          const SizedBox(height: 4),
          Text(val, style: TextStyle(fontSize: 20, fontWeight: FontWeight.w900, color: col)),
          Text(status, style: TextStyle(fontSize: 9, fontWeight: FontWeight.bold, color: col.withOpacity(0.7))),
        ],
      ),
    );
  }
}