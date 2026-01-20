import 'dart:convert';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:fl_chart/fl_chart.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:countries_world_map/countries_world_map.dart';
import 'package:flutter_staggered_animations/flutter_staggered_animations.dart';

void main() {
  runApp(const AadhaarDarpanApp());
}

class AadhaarDarpanApp extends StatelessWidget {
  const AadhaarDarpanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Aadhaar Darpan | RGIPT NIU',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFD4AF37),
          brightness: Brightness.dark,
        ),
        textTheme: GoogleFonts.interTextTheme(ThemeData.dark().textTheme),
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

  // --- ENSURE THIS IP MATCHES YOUR PC'S IPv4 ---
  final String baseUrl = "http://192.168.56.211:5001/api";

  @override
  void initState() {
    super.initState();
    _fetchMetadata();
    _fetchHeatmap();
  }

  // --- DATA METHODS ---

  Future<void> _fetchMetadata() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/metadata')).timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        setState(() => metadata = json.decode(response.body)['metadata']);
      }
    } catch (e) {
      _showSnackbar("CONNECTION ERROR: VERIFY SERVER IP");
    }
  }

  Future<void> _fetchHeatmap() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/heatmap'));
      if (response.statusCode == 200) {
        setState(() => heatmapData = json.decode(response.body)['data']);
      }
    } catch (e) {
      debugPrint("Map Sync Error: $e");
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
      _showSnackbar("SYNC FAILED: CHECK FIREWALL");
    } finally {
      setState(() => isLoading = false);
    }
  }

  void _showSnackbar(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg, style: GoogleFonts.orbitron(fontSize: 10, fontWeight: FontWeight.bold)),
      backgroundColor: const Color(0xFFD4AF37),
      behavior: SnackBarBehavior.floating,
    ));
  }

  // --- UI COMPONENTS ---

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => FocusScope.of(context).unfocus(),
      child: Scaffold(
        extendBodyBehindAppBar: true,
        appBar: _buildAppBar(),
        body: Stack(
          children: [
            _buildBackgroundGradient(),
            _buildDataParticleOverlay(),
            SafeArea(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: 20.0),
                child: Column(
                  children: [
                    const SizedBox(height: 10),
                    _buildGlassPanel(_buildSelectionDropdowns(), glow: const Color(0xFFD4AF37)),
                    const SizedBox(height: 24),
                    _buildMainContent(),
                    const SizedBox(height: 40),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  PreferredSizeWidget _buildAppBar() {
    return AppBar(
      title: Column(
        children: [
          Text("AADHAAR DARPAN", style: GoogleFonts.spaceGrotesk(letterSpacing: 4, fontWeight: FontWeight.w900, color: Colors.white)),
          Text("RGIPT NATIONAL INTELLIGENCE UNIT", style: GoogleFonts.orbitron(fontSize: 7, color: const Color(0xFFD4AF37), letterSpacing: 1.2)),
        ],
      ),
      backgroundColor: Colors.transparent,
      elevation: 0,
      centerTitle: true,
      flexibleSpace: ClipRect(child: BackdropFilter(filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10), child: Container(color: Colors.black12))),
    );
  }

  Widget _buildBackgroundGradient() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF0A192F), Color(0xFF112240), Color(0xFF020C1B)],
        ),
      ),
    );
  }

  Widget _buildDataParticleOverlay() {
    return Opacity(
      opacity: 0.05,
      child: GridView.builder(
        physics: const NeverScrollableScrollPhysics(),
        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 15),
        itemBuilder: (context, index) => Center(child: Container(width: 1, height: 1, decoration: const BoxDecoration(color: Colors.white, shape: BoxShape.circle))),
      ),
    );
  }

  Widget _buildMainContent() {
    if (isLoading) {
      return const Center(child: Padding(padding: EdgeInsets.all(50), child: CircularProgressIndicator(color: Color(0xFFD4AF37))));
    }
    if (auditData != null) {
      return _buildReportView();
    }
    return _buildHomeScreen();
  }

  Widget _buildSelectionDropdowns() {
    return Column(
      children: [
        DropdownButtonFormField<String>(
          isExpanded: true, // <--- ADDED THIS LINE (Fixes the overflow)
          dropdownColor: const Color(0xFF112240),
          style: GoogleFonts.inter(color: Colors.white, fontWeight: FontWeight.w600),
          decoration: _inputStyle("GEO-SPATIAL SCOPE (STATE)", Icons.language),
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
          isExpanded: true, // <--- ADDED THIS LINE (Safety for the second dropdown)
          dropdownColor: const Color(0xFF112240),
          style: GoogleFonts.inter(color: Colors.white, fontWeight: FontWeight.w600),
          decoration: _inputStyle("REGIONAL DRILLDOWN (DISTRICT)", Icons.gps_fixed),
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
    );
  }

  Widget _buildHomeScreen() {
    if (heatmapData == null) return const Center(child: Text("Initializing Neural Audit...", style: TextStyle(color: Colors.white24, letterSpacing: 2)));
    return Column(
      children: [
        Text("NATIONAL AUDIT HEATMAP", style: GoogleFonts.orbitron(letterSpacing: 4, fontSize: 11, color: Colors.white70)),
        const SizedBox(height: 25),
        // FIXED: Container with forced constraints and FittedBox to fix the glitch
        Container(
          height: 380,
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.1),
            borderRadius: BorderRadius.circular(30),
          ),
          child: FittedBox(
            fit: BoxFit.contain, // Prevents the map from stretching and glitching
            child: SizedBox(
              width: 500, // Fixed internal width for vector stability
              height: 600,
              child: SimpleMap(
                instructions: SMapIndia.instructions,
                defaultColor: Colors.white.withOpacity(0.05),
                colors: _generateNeonMapColors(),
                callback: (id, name, taparea) {
                  setState(() {
                    selectedState = name.toUpperCase();
                    selectedDistrict = null;
                    _fetchAuditReport();
                  });
                },
              ),
            ),
          ),
        ),
        const SizedBox(height: 20),
        _buildLegend(),
      ],
    );
  }

  Widget _buildReportView() {
    final cards = auditData!['cards'];
    final forecast = List<dynamic>.from(cards['efficiency']['biometric_traffic_trend']);
    final double modelAccuracy = (cards['efficiency']['accuracy'] ?? 94.2).toDouble();
    double maxVal = forecast.map((e) => (e as num).toDouble()).reduce((a, b) => a > b ? a : b);

    return AnimationLimiter(
      child: Column(
        children: AnimationConfiguration.toStaggeredList(
          duration: const Duration(milliseconds: 1000),
          childAnimationBuilder: (widget) => SlideAnimation(verticalOffset: 50.0, child: FadeInAnimation(child: widget)),
          children: [
            _buildSectionHeader("PILLAR I: CHILD ENROLMENT DEPTH", Icons.face_retouching_natural),
            const SizedBox(height: 12),
            _buildCoveragePillar(cards),
            const SizedBox(height: 24),
            _buildSectionHeader("PILLAR II: SYSTEM PERFORMANCE", Icons.speed),
            const SizedBox(height: 12),
            _buildPerformanceTiles(cards),
            const SizedBox(height: 24),
            _buildForecastHeader(modelAccuracy),
            const SizedBox(height: 12),
            _buildForecastChart(forecast, maxVal, cards),
            const SizedBox(height: 30),
            _buildActionButtons(),
          ],
        ),
      ),
    );
  }

  Widget _buildCoveragePillar(Map cards) {
    double coveredValue = (cards['inclusivity']['value'] as num).toDouble();
    return _buildGlassPanel(
      Column(
        children: [
          SizedBox(
            height: 200,
            child: Stack(
              alignment: Alignment.center,
              children: [
                PieChart(PieChartData(
                  sectionsSpace: 2,
                  centerSpaceRadius: 55,
                  sections: [
                    PieChartSectionData(
                      value: coveredValue,
                      color: const Color(0xFFD4AF37),
                      radius: 22,
                      showTitle: false,
                    ),
                    PieChartSectionData(
                      value: (100 - coveredValue).clamp(0, 100),
                      color: Colors.white.withOpacity(0.05),
                      radius: 18,
                      showTitle: false,
                    ),
                  ],
                )),
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text("${coveredValue.toStringAsFixed(1)}%", style: GoogleFonts.orbitron(fontSize: 22, fontWeight: FontWeight.bold, color: const Color(0xFFD4AF37))),
                    Text("ENROLLED", style: GoogleFonts.inter(fontSize: 9, color: Colors.white30, fontWeight: FontWeight.bold, letterSpacing: 1)),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _buildChartLegend(const Color(0xFFD4AF37), "Enrolled"),
              const SizedBox(width: 20),
              _buildChartLegend(Colors.white.withOpacity(0.1), "Pending"),
            ],
          ),
          const SizedBox(height: 16),
          Text("${cards['inclusivity']['status']}", style: GoogleFonts.orbitron(color: const Color(0xFFD4AF37), fontSize: 12, fontWeight: FontWeight.bold)),
          const Text("Regional Saturation Level", style: TextStyle(color: Colors.white30, fontSize: 9)),
        ],
      ),
      glow: const Color(0xFFD4AF37),
    );
  }

  Widget _buildChartLegend(Color color, String label) {
    return Row(children: [
      Container(width: 8, height: 8, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
      const SizedBox(width: 6),
      Text(label, style: const TextStyle(fontSize: 9, color: Colors.white54)),
    ]);
  }

  Widget _buildPerformanceTiles(Map cards) {
    return Row(
      children: [
        Expanded(child: _buildStatusTile("PROCESSING SPEED", "${cards['security']['value']}%", cards['security']['status'])),
        const SizedBox(width: 12),
        Expanded(child: _buildStatusTile("ENROLMENT RATIO", "${cards['inclusivity']['value']}%", cards['inclusivity']['status'])),
      ],
    );
  }

  Widget _buildForecastHeader(double accuracy) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        // FIXED: Wrapped in Expanded to prevents pushing the right-side badge off-screen
        Expanded(
          child: _buildSectionHeader("PILLAR III: 90-DAY LOAD PREDICTION", Icons.online_prediction),
        ),
        const SizedBox(width: 8), // Added spacing so text doesn't touch the badge
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
          decoration: BoxDecoration(color: Colors.white.withOpacity(0.05), borderRadius: BorderRadius.circular(10), border: Border.all(color: const Color(0xFFD4AF37), width: 0.5)),
          child: Text("AI CONFIDENCE: $accuracy%", style: GoogleFonts.robotoMono(fontSize: 9, color: const Color(0xFFD4AF37), fontWeight: FontWeight.bold)),
        ),
      ],
    );
  }

  Widget _buildForecastChart(List forecast, double maxVal, Map cards) {
    return _buildGlassPanel(Column(
      children: [
        SizedBox(
          height: 220,
          child: BarChart(BarChartData(
            alignment: BarChartAlignment.spaceAround,
            maxY: maxVal * 1.3,
            barGroups: List.generate(3, (i) => BarChartGroupData(x: i, barRods: [
              BarChartRodData(
                toY: (forecast[i] as num).toDouble(),
                color: const Color(0xFFD4AF37),
                width: 45,
                borderRadius: const BorderRadius.vertical(top: Radius.circular(8)),
                backDrawRodData: BackgroundBarChartRodData(show: true, toY: maxVal * 1.3, color: Colors.white.withOpacity(0.02)),
              )
            ])),
            titlesData: FlTitlesData(
              show: true,
              bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, getTitlesWidget: (val, _) {
                const labels = ['MONTH 1', 'MONTH 2', 'MONTH 3'];
                return Padding(padding: const EdgeInsets.only(top: 10), child: Text(labels[val.toInt()], style: const TextStyle(color: Colors.white38, fontSize: 8)));
              })),
              leftTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
              topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
              rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            ),
            gridData: const FlGridData(show: false),
            borderData: FlBorderData(show: false),
          )),
        ),
        const SizedBox(height: 15),
        Text("AI PREDICTION TREND: ${cards['efficiency']['trend']}", style: GoogleFonts.orbitron(fontSize: 9, color: Colors.white54, letterSpacing: 2)),
      ],
    ));
  }

  // --- REUSABLE BUILDERS ---

  Widget _buildGlassPanel(Widget child, {Color glow = Colors.white12}) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 30, sigmaY: 30),
        child: Container(
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.03),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: glow.withOpacity(0.2), width: 0.8),
          ),
          child: child,
        ),
      ),
    );
  }

  Widget _buildStatusTile(String title, String val, String status) {
    Color col = status.contains("OPTIMIZED") || status.contains("EASY") ? const Color(0xFF64FFDA) : (status.contains("IMPROVING") || status.contains("MODERATE") ? Colors.white70 : const Color(0xFFF05454));
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(color: Colors.white.withOpacity(0.03), borderRadius: BorderRadius.circular(20), border: Border.all(color: col.withOpacity(0.2))),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: GoogleFonts.orbitron(fontSize: 7, color: Colors.white24)),
          const SizedBox(height: 8),
          Text(val, style: GoogleFonts.robotoMono(fontSize: 18, fontWeight: FontWeight.bold, color: col)),
          const SizedBox(height: 4),
          Text(status, style: TextStyle(fontSize: 8, color: col.withOpacity(0.7), fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildSectionHeader(String title, IconData icon) {
    return Row(children: [
      Icon(icon, size: 16, color: const Color(0xFFD4AF37)),
      const SizedBox(width: 8),
      // FIXED: Wrapped in Flexible so the text wraps nicely if space is tight
      Flexible(
        child: Text(title, style: GoogleFonts.orbitron(fontSize: 9, color: Colors.white54, fontWeight: FontWeight.w800, letterSpacing: 1.5)),
      ),
    ]);
  }

  Widget _buildActionButtons() {
    return Column(
      children: [
        TextButton.icon(
          onPressed: () => setState(() => auditData = null),
          icon: const Icon(Icons.keyboard_backspace, color: Color(0xFFD4AF37)),
          label: Text("BACK TO NATIONAL OVERVIEW", style: GoogleFonts.orbitron(color: const Color(0xFFD4AF37), fontSize: 10, fontWeight: FontWeight.bold)),
        ),
        const SizedBox(height: 20),
        Text("RGIPT INTEL UNIT | OFFICIAL PROJECT", style: GoogleFonts.orbitron(fontSize: 7, color: Colors.white10, letterSpacing: 2)),
      ],
    );
  }

  InputDecoration _inputStyle(String label, IconData icon) {
    return InputDecoration(
      labelText: label,
      labelStyle: GoogleFonts.orbitron(color: const Color(0xFFD4AF37), fontSize: 8, letterSpacing: 1),
      prefixIcon: Icon(icon, color: const Color(0xFFD4AF37), size: 18),
      filled: true,
      fillColor: Colors.black26,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(15), borderSide: BorderSide.none),
    );
  }

  Map<String, Color> _generateNeonMapColors() {
    Map<String, Color> colors = {};
    heatmapData?.forEach((state, details) {
      String status = details['status'];
      colors[state] = status == "SAFE" ? const Color(0xFF64FFDA).withOpacity(0.6) : (status == "WARNING" ? Colors.white70 : const Color(0xFFF05454));
    });
    return colors;
  }

  Widget _buildLegend() {
    return Row(mainAxisAlignment: MainAxisAlignment.center, children: [
      _legendNode("CRITICAL", const Color(0xFFF05454)),
      const SizedBox(width: 20),
      _legendNode("NEEDS AUDIT", Colors.white60),
      const SizedBox(width: 20),
      _legendNode("STABLE", const Color(0xFF64FFDA)),
    ]);
  }

  Widget _legendNode(String label, Color color) {
    return Row(children: [
      Container(width: 6, height: 6, decoration: BoxDecoration(color: color, shape: BoxShape.circle, boxShadow: [BoxShadow(color: color, blurRadius: 10)])),
      const SizedBox(width: 8),
      Text(label, style: GoogleFonts.orbitron(fontSize: 8, color: Colors.white30)),
    ]);
  }
}