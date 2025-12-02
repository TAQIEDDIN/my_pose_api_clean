"""
Sağlık standartlarına dayalı tıbbi kurallar
"""

class MedicalRules:
    """Durum analizi için tıbbi kurallar"""
    
    # Tıbbi eşikler (mm cinsinden)
    THRESHOLDS = {
        'normal': 3.0,      # 3 mm'den az = Normal
        'mild': 8.0,        # 3-8 mm = Hafif
        'moderate': 15.0,   # 8-15 mm = Orta
        'severe': float('inf')  # 15 mm'den fazla = Şiddetli
    }
    
    @staticmethod
    def classify_severity(shoulder_diff, hip_diff):
        """Risk seviyesini sınıflandır"""
        max_diff = max(shoulder_diff, hip_diff)
        
        if max_diff < MedicalRules.THRESHOLDS['normal']:
            return {
                'level': 'normal',
                'label': 'Normal',
                'color': 'green',
                'icon': '✅',
                'urgency': 'Yok'
            }
        elif max_diff < MedicalRules.THRESHOLDS['mild']:
            return {
                'level': 'mild',
                'label': 'Hafif',
                'color': 'yellow',
                'icon': '⚠️',
                'urgency': 'Rutin takip'
            }
        elif max_diff < MedicalRules.THRESHOLDS['moderate']:
            return {
                'level': 'moderate',
                'label': 'Orta',
                'color': 'orange',
                'icon': '🔶',
                'urgency': '2 hafta içinde doktor kontrolü'
            }
        else:
            return {
                'level': 'severe',
                'label': 'Şiddetli',
                'color': 'red',
                'icon': '🔴',
                'urgency': '1 hafta içinde acil kontrol'
            }
    
    @staticmethod
    def identify_issues(shoulder_diff, hip_diff, shoulder_side, hip_side):
        """Başlıca sorunları belirle"""
        issues = []
        
        if shoulder_diff >= 3:
            issues.append({
                'type': 'shoulder_imbalance',
                'severity': 'mild' if shoulder_diff < 8 else 'moderate' if shoulder_diff < 15 else 'severe',
                'value': shoulder_diff,
                'side': shoulder_side,
                'description': f'Omuz dengesizliği: fark {shoulder_diff:.1f} mm'
            })
        
        if hip_diff >= 3:
            issues.append({
                'type': 'hip_imbalance',
                'severity': 'mild' if hip_diff < 8 else 'moderate' if hip_diff < 15 else 'severe',
                'value': hip_diff,
                'side': hip_side,
                'description': f'Kalça dengesizliği: fark {hip_diff:.1f} mm'
            })
        
        # Denge bozukluğu paterni
        if len(issues) >= 2:
            shoulder_issue = next((i for i in issues if i['type'] == 'shoulder_imbalance'), None)
            hip_issue = next((i for i in issues if i['type'] == 'hip_imbalance'), None)
            
            if shoulder_issue and hip_issue:
                # Aynı taraf
                if 'Left' in shoulder_issue['side'] and 'Left' in hip_issue['side']:
                    issues.append({
                        'type': 'pattern',
                        'severity': 'moderate',
                        'description': 'Yan patern bozukluğu: Sol taraf tamamen yüksek'
                    })
                elif 'Right' in shoulder_issue['side'] and 'Right' in hip_issue['side']:
                    issues.append({
                        'type': 'pattern',
                        'severity': 'moderate',
                        'description': 'Yan patern bozukluğu: Sağ taraf tamamen yüksek'
                    })
                else:
                    issues.append({
                        'type': 'pattern',
                        'severity': 'moderate',
                        'description': 'Telafi edici karşıt patern: Vücut dengeyi sağlamaya çalışıyor'
                    })
        
        return issues
    
    @staticmethod
    def predict_health_impacts(severity_level, issues):
        """Sağlık etkilerini tahmin et"""
        impacts = []
        
        if severity_level == 'normal':
            impacts.append({
                'category': 'general',
                'impact': 'Beklenen sağlık etkisi yok',
                'probability': 'Çok düşük'
            })
        
        elif severity_level == 'mild':
            has_shoulder = any(i['type'] == 'shoulder_imbalance' for i in issues)
            has_hip = any(i['type'] == 'hip_imbalance' for i in issues)
            
            if has_shoulder:
                impacts.extend([
                    {
                        'category': 'musculoskeletal',
                        'impact': 'Boyun ve omuzda hafif kas gerginliği',
                        'probability': 'Orta',
                        'timeframe': 'Haftalar'
                    },
                    {
                        'category': 'neurological',
                        'impact': 'Ara sıra gerilim tipi baş ağrısı',
                        'probability': 'Düşük',
                        'timeframe': 'Aylar'
                    }
                ])
            
            if has_hip:
                impacts.extend([
                    {
                        'category': 'musculoskeletal',
                        'impact': 'Uzun süre ayakta kalmada bel yorgunluğu',
                        'probability': 'Orta',
                        'timeframe': 'Haftalar'
                    },
                    {
                        'category': 'functional',
                        'impact': 'Yürürken daha çabuk yorulma',
                        'probability': 'Düşük',
                        'timeframe': 'Aylar'
                    }
                ])
        
        elif severity_level == 'moderate':
            impacts.extend([
                {
                    'category': 'musculoskeletal',
                    'impact': 'Tekrarlayan boyun ve omuz ağrısı',
                    'probability': 'Yüksek',
                    'timeframe': 'Günler-Haftalar'
                },
                {
                    'category': 'musculoskeletal',
                    'impact': 'Keskin bel ağrısı',
                    'probability': 'Orta-Yüksek',
                    'timeframe': 'Haftalar'
                },
                {
                    'category': 'neurological',
                    'impact': 'Özellikle akşamları sık baş ağrısı',
                    'probability': 'Orta',
                    'timeframe': 'Haftalar'
                },
                {
                    'category': 'sleep',
                    'impact': 'Ağrı nedeniyle uyku zorluğu',
                    'probability': 'Orta',
                    'timeframe': 'Haftalar-Aylar'
                },
                {
                    'category': 'structural',
                    'impact': 'Hafif skolyoz gelişme ihtimali',
                    'probability': 'Düşük',
                    'timeframe': 'Yıllar'
                }
            ])
        
        elif severity_level == 'severe':
            impacts.extend([
                {
                    'category': 'musculoskeletal',
                    'impact': 'Şiddetli ve sürekli boyun-bel ağrısı',
                    'probability': 'Çok yüksek',
                    'timeframe': 'Günlük'
                },
                {
                    'category': 'structural',
                    'impact': 'Skolyoz (omurga yan eğriliği)',
                    'probability': 'Orta-Yüksek',
                    'timeframe': 'Aylar-Yıllar'
                },
                {
                    'category': 'neurological',
                    'impact': 'Sinir basısı ve uyuşma ihtimali',
                    'probability': 'Orta',
                    'timeframe': 'Aylar'
                },
                {
                    'category': 'functional',
                    'impact': 'Hareket ve günlük aktivite kısıtlamaları',
                    'probability': 'Yüksek',
                    'timeframe': 'Haftalar'
                },
                {
                    'category': 'degenerative',
                    'impact': 'Erken disk dejenerasyonu',
                    'probability': 'Orta',
                    'timeframe': 'Yıllar'
                },
                {
                    'category': 'respiratory',
                    'impact': 'Nefes problemleri (çok şiddetli durumlarda)',
                    'probability': 'Düşük',
                    'timeframe': 'Yıllar'
                }
            ])
        
        return impacts
    
    @staticmethod
    def recommend_exercises(severity_level, issues):
        """Uygun egzersiz önerileri"""
        exercises = {
            'shoulder': [],
            'hip': [],
            'general': []
        }
        
        has_shoulder = any(i['type'] == 'shoulder_imbalance' for i in issues)
        has_hip = any(i['type'] == 'hip_imbalance' for i in issues)
        
        if severity_level == 'normal':
            exercises['general'] = [
                {
                    'name': 'Günlük yürüyüş',
                    'duration': '30 dakika',
                    'frequency': 'Her gün',
                    'difficulty': 'Kolay'
                },
                {
                    'name': 'Genel esneme egzersizleri',
                    'duration': '10 dakika',
                    'frequency': 'Sabah ve akşam',
                    'difficulty': 'Kolay'
                }
            ]
        
        elif severity_level in ['mild', 'moderate']:
            if has_shoulder:
                exercises['shoulder'] = [
                    {
                        'name': 'Omuz geriye çekme (Scapular Retraction)',
                        'sets': '3 set',
                        'reps': '15 tekrar',
                        'hold': '5 saniye',
                        'frequency': 'Günde 2 kez',
                        'notes': 'Dik oturun, kürek kemiklerini geriye ve aşağı çekin'
                    },
                    {
                        'name': 'Göğüs esnetme (Doorway Stretch)',
                        'sets': 'Her iki taraf',
                        'hold': '30 saniye',
                        'frequency': 'Günde 3 kez',
                        'notes': 'Kapı kenarında 90° açıyla kolunuzu yerleştirin'
                    },
                    {
                        'name': 'Duvar melekleri (Wall Angels)',
                        'sets': '3 set',
                        'reps': '10 tekrar',
                        'frequency': 'Her gün',
                        'notes': 'Sırtınızı duvara yaslayın, kollarınızı yukarı-aşağı hareket ettirin'
                    }
                ]
            
            if has_hip:
                exercises['hip'] = [
                    {
                        'name': 'Köprü egzersizi (Glute Bridge)',
                        'sets': '3 set',
                        'reps': '12-15 tekrar',
                        'hold': '5 saniye yukarıda',
                        'frequency': 'Her gün',
                        'notes': 'Sırt üstü uzan, dizleri bük, kalçanı kaldır'
                    },
                    {
                        'name': 'Clamshells',
                        'sets': 'Her taraf için 3 set',
                        'reps': '15 tekrar',
                        'frequency': 'Her gün',
                        'notes': 'Yan yat, dizini aç ayakları birlikte tut'
                    },
                    {
                        'name': 'Kalça fleksör esnetme (Hip Flexor Stretch)',
                        'sets': 'Her iki taraf',
                        'hold': '30 saniye',
                        'frequency': 'Günde 2 kez',
                        'notes': 'Bir diz üstünde dur, kalçanı öne doğru it'
                    }
                ]
            
            exercises['general'] = [
                {
                    'name': 'Kedi-İnek (Cat-Cow)',
                    'sets': '2 set',
                    'reps': '10 tekrar',
                    'frequency': 'Sabah ve akşam',
                    'notes': 'Eller ve dizler üzerinde sıranı eğ ve kambur yap'
                },
                {
                    'name': 'Kuş-Köpek (Bird Dog)',
                    'sets': '3 set',
                    'reps': 'Her taraf için 10 tekrar',
                    'frequency': 'Her gün',
                    'notes': 'Zıt kol ve bacağı uzat, dengenizi koru'
                }
            ]
        
        elif severity_level == 'severe':
            exercises['general'] = [
                {
                    'name': 'Uyarı',
                    'notes': '⚠️ Fizik tedavi uzmanına danışmadan egzersize başlamayın',
                    'reason': 'Durum profesyonel değerlendirme gerektiriyor'
                },
                {
                    'name': 'Geçici hafif egzersizler',
                    'notes': 'Sadece çok hafif esneme hareketleri, doktora gidene kadar'
                }
            ]
        
        return exercises
    
    @staticmethod
    def get_medical_advice(severity_level):
        """Risk seviyesine göre tıbbi öneriler"""
        advice = {
            'normal': {
                'consultation': 'Şu an gerekli değil',
                'follow_up': 'Yıllık rutin kontrol',
                'emergency_signs': ['Ani şiddetli ağrı', 'Uçlarda uyuşma']
            },
            'mild': {
                'consultation': 'Fizik tedavi uzmanıyla isteğe bağlı görüşme',
                'follow_up': '3 ay sonra yeniden değerlendirme',
                'emergency_signs': ['2 haftadan uzun süren ağrı', 'Belirtilerin kötüleşmesi']
            },
            'moderate': {
                'consultation': '2 hafta içinde fizik tedavi uzmanına git',
                'follow_up': 'Her 4 haftada bir takip',
                'emergency_signs': ['Dayanılmaz şiddetli ağrı', 'Sürekli uyuşma', 'Hareket zorluğu']
            },
            'severe': {
                'consultation': '🚨 1 hafta içinde ortopedi uzmanına acil ziyaret',
                'tests_needed': ['Röntgen', 'Bacak uzunluğu ölçümü', 'Kas gücü değerlendirmesi'],
                'follow_up': 'Haftalık takip',
                'emergency_signs': ['Dayanılmaz ağrı', 'Mesane kontrol kaybı', 'Bacaklarda şiddetli güçsüzlük']
            }
        }
        
        return advice.get(severity_level, advice['normal'])
