"""
Ana analizör - tüm modelleri ve kuralları entegre eder
"""
from .medical_rules import MedicalRules
from .report_generator import ReportGenerator

class PostureAnalyzer:
    """Duruş için kapsamlı analizör"""
    
    def __init__(self):
        self.rules = MedicalRules()
        self.report_gen = ReportGenerator()
    
    def analyze(self, analysis_data):
        """
        Duruş verilerinin kapsamlı analizi
        
        Args:
            analysis_data: sözlük içerir:
                - shoulder_y_diff: omuzlar arasındaki fark (mm)
                - hip_y_diff: kalçalar arasındaki fark (mm)
                - shoulder_higher_side: daha yüksek olan omuz tarafı
                - hip_higher_side: daha yüksek olan kalça tarafı
                - angles: vücut açıları (opsiyonel)
        
        Returns:
            dict: tam analiz sonuçları
        """
        # Verileri çıkar
        shoulder_diff = analysis_data.get('shoulder_y_diff', 0) or 0
        hip_diff = analysis_data.get('hip_y_diff', 0) or 0
        shoulder_side = analysis_data.get('shoulder_higher_side', 'Bilinmiyor')
        hip_side = analysis_data.get('hip_higher_side', 'Bilinmiyor')
        angles = analysis_data.get('angles', {})
        
        # 1. Şiddeti sınıflandır
        severity = self.rules.classify_severity(shoulder_diff, hip_diff)
        
        # 2. Problemleri belirle
        issues = self.rules.identify_issues(
            shoulder_diff, hip_diff, 
            shoulder_side, hip_side
        )
        
        # 3. Sağlık etkilerini tahmin et
        health_impacts = self.rules.predict_health_impacts(
            severity['level'], issues
        )
        
        # 4. Egzersiz önerileri
        exercises = self.rules.recommend_exercises(
            severity['level'], issues
        )
        
        # 5. Tıbbi tavsiyeler
        medical_advice = self.rules.get_medical_advice(severity['level'])
        
        # 6. Sonuçları birleştir
        result = {
            'shoulder_diff': shoulder_diff,
            'hip_diff': hip_diff,
            'shoulder_side': shoulder_side,
            'hip_side': hip_side,
            'severity': severity,
            'issues': issues,
            'health_impacts': health_impacts,
            'exercises': exercises,
            'medical_advice': medical_advice,
            'angles': angles
        }
        
        return result
    
    def generate_report(self, analysis_data):
        """
        Tam analiz + rapor oluşturma
        
        Args:
            analysis_data: analiz verileri
        
        Returns:
            str: tam Markdown raporu
        """
        # Analizi yap
        result = self.analyze(analysis_data)
        
        # Rapor oluştur
        report = self.report_gen.generate_full_report(result)
        
        return report
    
    def get_quick_summary(self, analysis_data):
        """
        Hızlı durum özeti (arayüz için)
        
        Returns:
            dict: kısa özet
        """
        result = self.analyze(analysis_data)
        
        return {
            'severity_level': result['severity']['level'],
            'severity_label': result['severity']['label'],
            'severity_icon': result['severity']['icon'],
            'urgency': result['severity']['urgency'],
            'num_issues': len(result['issues']),
            'main_issue': result['issues'][0]['description'] if result['issues'] else 'Herhangi bir sorun yok',
            'recommendation': self._get_short_recommendation(result['severity']['level'])
        }
    
    def _get_short_recommendation(self, severity):
        """Kısa öneri"""
        recommendations = {
            'normal': 'Sağlıklı yaşam tarzına devam edin',
            'mild': 'Basit düzeltici egzersizlere başlayın',
            'moderate': 'Yakında bir fizyoterapiste danışın',
            'severe': 'Acil tıbbi muayene gerekli'
        }
        return recommendations.get(severity, '')
