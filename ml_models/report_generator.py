"""
Akıllı tıbbi rapor oluşturucu
"""
from datetime import datetime

class ReportGenerator:
    """Ayrıntılı tıbbi rapor üretimi"""
    
    @staticmethod
    def generate_full_report(analysis_result):
        """
        Analiz sonuçlarına dayalı tam rapor oluşturma
        
        Args:
            analysis_result: analiz sonuçlarını içeren sözlük
        
        Returns:
            str: Markdown formatında rapor
        """
        sections = []
        
        # Rapor başlığı
        sections.append(ReportGenerator._generate_header(analysis_result))
        
        # Genel değerlendirme
        sections.append(ReportGenerator._generate_assessment(analysis_result))
        
        # Tespit edilen problemler
        sections.append(ReportGenerator._generate_issues(analysis_result))
        
        # Sağlık etkileri
        sections.append(ReportGenerator._generate_health_impacts(analysis_result))
        
        # Önerilen egzersizler
        sections.append(ReportGenerator._generate_exercises(analysis_result))
        
        # 💡 AttributeError hatasını çözmek için eklendi
        sections.append(ReportGenerator._generate_daily_tips(analysis_result))
        
        # Tıbbi öneriler
        sections.append(ReportGenerator._generate_medical_advice(analysis_result))
        
        # Alt bilgi
        sections.append(ReportGenerator._generate_footer())
        
        return "\n\n".join(sections)
    
    @staticmethod
    def _generate_header(result):
        """Rapor başlığı"""
        severity = result['severity']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        return f"""# 📋 Duruş Analizi Tıbbi Raporu

Tarih ve Saat: {timestamp}  
Durum Seviyesi: {severity['icon']} {severity['label']}  
Aciliyet Derecesi: {severity['urgency']}

---"""
    
    @staticmethod
    def _generate_assessment(result):
        """Genel değerlendirme"""
        severity = result['severity']
        shoulder = result['shoulder_diff']
        hip = result['hip_diff']
        
        assessment = "## 🔍 Genel Değerlendirme\n\n"
        
        if severity['level'] == 'normal':
            assessment += f"""{severity['icon']} Mükemmel duruş!

Vücudunuz çok iyi bir denge halinde. Ölçülen farklılıklar:
- Omuzlar: {shoulder:.1f} mm
- Kalçalar: {hip:.1f} mm

Bu farklılıklar tamamen normaldir ve endişelenmeye gerek yok. Sağlıklı yaşam tarzınıza devam edin."""

        elif severity['level'] == 'mild':
            assessment += f"""{severity['icon']} Kolayca düzeltilebilecek küçük dengesizlikler

Bazı küçük dengesizlikler tespit edildi:
- Omuz farkı: {shoulder:.1f} mm
- Kalça farkı: {hip:.1f} mm

Bu dengesizlikler yaygındır ve düzenli egzersiz ve duruş farkındalığı ile kolayca düzeltilebilir."""

        elif severity['level'] == 'moderate':
            assessment += f"""{severity['icon']} Orta düzeyde dengesizlikler - dikkat gerektirir

Belirgin dengesizlikler bulundu:
- Omuz farkı: {shoulder:.1f} mm
- Kalça farkı: {hip:.1f} mm

Bu dengesizlikler düzenli bir tedavi programı gerektirir. Kişisel bir düzeltme planı için fizyoterapiste başvurmanız önerilir."""

        else:  # severe
            assessment += f"""{severity['icon']} Uyarı: Ciddi dengesizlikler

⚠️ Çok önemli: Acil tıbbi değerlendirme gerektiren ciddi dengesizlikler bulundu:
- Omuz farkı: {shoulder:.1f} mm
- Kalça farkı: {hip:.1f} mm

Bu düzeydeki dengesizlik yapısal bir soruna işaret edebilir. Lütfen tıbbi danışmayı ertelemeyin."""

        return assessment
    
    @staticmethod
    def _generate_issues(result):
        """Tespit edilen problemler"""
        issues = result['issues']
        
        if not issues:
            return "## ✅ Tespit Edilen Problemler\n\nHerhangi bir problem bulunamadı."
        
        section = "## 🔎 Tespit Edilen Problemler\n\n"
        
        for issue in issues:
            if issue['type'] == 'shoulder_imbalance':
                section += f"""### Omuzlar
- Problem: {issue['description']}
- Etkilenen taraf: {issue['side']}
- Ciddiyet: {issue['severity']}

"""
            elif issue['type'] == 'hip_imbalance':
                section += f"""### Kalçalar
- Problem: {issue['description']}
- Etkilenen taraf: {issue['side']}
- Ciddiyet: {issue['severity']}

"""
            elif issue['type'] == 'pattern':
                section += f"""### Dengesizlik Deseni
⚠️ Önemli Not: {issue['description']}

Bu desen şunlara işaret edebilir:
- Vücudun belli bir duruşa alışması
- Bacak boyu farklılığı ihtimali
- Eski bir yaralanmaya karşı vücudun telafisi

"""
        
        return section.rstrip()
    
    @staticmethod
    def _generate_health_impacts(result):
        """Sağlık etkileri"""
        impacts = result['health_impacts']
        
        if not impacts:
            return ""
        
        section = "## 🏥 Olası Sağlık Etkileri\n\n"
        
        # Kategoriye göre grupla
        by_category = {}
        for impact in impacts:
            cat = impact['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(impact)
        
        category_names = {
            'general': 'Genel',
            'musculoskeletal': 'Kas-iskelet sistemi',
            'neurological': 'Sinir sistemi',
            'functional': 'Fonksiyonel kapasite',
            'sleep': 'Uyku',
            'structural': 'Yapısal',
            'degenerative': 'Dejeneratif',
            'respiratory': 'Solunum sistemi'
        }
        
        for cat, items in by_category.items():
            section += f" {category_names.get(cat, cat)}\n\n"
            for item in items:
                prob_emoji = '🔴' if item['probability'] == 'Yüksek' else '🟡' if item['probability'] == 'Orta' else '🟢'
                section += f"""- {prob_emoji} {item['impact']}
  - Olasılık: {item['probability']}
  - Zaman dilimi: {item.get('timeframe', 'Belirtilmemiş')}

"""
        
        return section.rstrip()
    
    @staticmethod
    def _generate_exercises(result):
        """Önerilen egzersizler"""
        exercises = result['exercises']
        severity = result['severity']['level']
        
        if severity == 'severe':
            return """ ⚠️ Egzersizler

Önemli Uyarı: Fizyoterapist ile görüşmeden herhangi bir egzersiz programına başlamayın.

Görüşme gününe kadar şunları yapabilirsiniz:
- Hafif yürüyüş (10-15 dakika)
- Çok hafif boyun esneme hareketleri
- Ağırlık kaldırmaktan tamamen kaçının"""
        
        section = "  Önerilen Egzersiz Programı\n\n"
        
        if exercises['shoulder']:
            section += " Omuzlar ve üst vücut için egzersizler\n\n"
            for ex in exercises['shoulder']:
                section += f""" {ex['name']}
- Setler: {ex.get('sets', 'N/A')}
- Tekrarlar: {ex.get('reps', 'N/A')}
- Bekleme: {ex.get('hold', 'Yapabildiğiniz kadar')}
- Sıklık: {ex['frequency']}
- Notlar: {ex.get('notes', '')}

"""
        
        if exercises['hip']:
            section += " Kalçalar ve alt vücut için egzersizler\n\n"
            for ex in exercises['hip']:
                section += f""" {ex['name']}
- Setler: {ex.get('sets', 'N/A')}
- Tekrarlar: {ex.get('reps', 'N/A')}
- Bekleme: {ex.get('hold', 'Yapabildiğiniz kadar')}
- Sıklık: {ex['frequency']}
- Notlar: {ex.get('notes', '')}

"""
        
        if exercises['general']:
            section += " Genel denge egzersizleri\n\n"
            for ex in exercises['general']:
                if ex.get('name') == 'Uyarı':
                    section += f" {ex.get('notes', '')}\n\n"
                    continue
                    
                section += f""" {ex['name']}
- Setler: {ex.get('sets', 'Yapabildiğiniz kadar')}
- Tekrarlar: {ex.get('reps', ex.get('duration', 'Dinlenmeye göre'))}
- Sıklık: {ex.get('frequency', 'Günlük')}
- Notlar: {ex.get('notes', '')}

"""
        
        return section.rstrip()
    
    @staticmethod
    def _generate_daily_tips(result):
        """Günlük duruş tavsiyeleri (AttributeError hatasını çözmek için eklendi)"""
        severity = result['severity']['level']
        
        if severity == 'severe':
            return ""  # Ciddi durumlarda günlük tavsiye yok (öncelik doktora yönlendirme)

        section = " Günlük Tavsiyeler\n\n"
        tips = [
            "Uzun süre 'text neck' (boyun öne eğilme) pozisyonunda oturmaktan kaçının.",
            "Ekran yüksekliğini gözleriniz ekranın üst üçte birlik kısmına gelecek şekilde ayarlayın.",
            "Her 30-60 dakikada bir kalkıp esneme yapın.",
            "Sırtüstü veya yan yatın, dizleriniz arasına yastık koyarak uyumaya çalışın."
        ]
        
        for tip in tips:
            section += f"- {tip}\n"
            
        return section.rstrip()

    @staticmethod
    def _generate_medical_advice(result):
        """Ciddiyet seviyesine göre tıbbi tavsiyeler"""
        severity = result['severity']['level']
        advice = result['medical_advice']
        
        section = "Tıbbi Tavsiyeler ve Önlemler\n\n"
        
        section += f"Danışmanlık: {advice.get('consultation', 'N/A')}\n\n"
        
        if advice.get('tests_needed'):
             section += f" Önerilen Tetkikler\n\n"
             for test in advice['tests_needed']:
                 section += f"- {test}\n"
        
        if advice.get('follow_up'):
            section += f" Takip\n\n"
            section += f"- {advice['follow_up']}\n"

        section += f"\nAcil durum belirtileri (görülürse doktora gitmeyi ertelemeyin):\n"
        for sign in advice['emergency_signs']:
             section += f"- 🔴 {sign}\n"

        return section.rstrip()

    @staticmethod
    def _generate_footer():
        """Rapor alt bilgisi"""
        return """---

Sorumluluk Reddi:** Bu rapor yalnızca analitik amaçlıdır ve uzman hekim tarafından yapılacak muayenenin yerine geçmez. Tanının doğrulanması ve uygun tedavi için doktor veya fizyoterapiste başvurulmalıdır."""
