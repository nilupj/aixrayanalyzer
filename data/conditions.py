def get_condition_info(condition):
    """
    Get educational information about a medical condition found in X-rays
    
    Args:
        condition (str): The name of the condition
        
    Returns:
        dict: Dictionary containing information about the condition
    """
    conditions_info = {
        # Bone-specific conditions
        "Fracture": {
            "description": "A break in a bone that occurs when physical force exceeds the bone's strength. Fractures can range from hairline cracks to complete breaks where the bone fragments separate.",
            "symptoms": "Pain that worsens with movement, swelling, bruising, deformity, limited mobility of the affected area, and sometimes audible cracking during the injury.",
            "treatment": "Treatment typically involves immobilizing the bone with a cast or brace to allow proper healing. Some fractures require surgical intervention with pins, plates, or screws to stabilize the bone. Physical therapy is often recommended during recovery."
        },
        "Osteoarthritis": {
            "description": "A degenerative joint disease characterized by the breakdown of joint cartilage and underlying bone, leading to joint space narrowing visible on X-rays.",
            "symptoms": "Joint pain, stiffness (especially in the morning), decreased range of motion, joint tenderness, and bone spurs that may be felt as hard lumps around the joint.",
            "treatment": "Management includes physical therapy, weight management, pain relievers, anti-inflammatory medications, and in severe cases, surgical joint replacement."
        },
        "Osteoporosis": {
            "description": "A condition characterized by decreased bone density, making bones more porous and fragile, increasing the risk of fractures.",
            "symptoms": "Often called a 'silent disease' as there may be no symptoms until a fracture occurs. Height loss and stooped posture may develop over time.",
            "treatment": "Treatment includes calcium and vitamin D supplements, weight-bearing exercise, and medications to slow bone loss and increase bone density."
        },
        "Scoliosis": {
            "description": "An abnormal sideways curvature of the spine that appears as an 'S' or 'C' shape rather than a straight line when viewed from behind on X-rays.",
            "symptoms": "Uneven shoulders or hips, one shoulder blade more prominent than the other, or a rotating spine. Severe cases may cause back pain and breathing difficulties.",
            "treatment": "Treatment depends on severity and may include observation, bracing (especially in growing children), or surgery for severe cases."
        },
        "Atelectasis": {
            "description": "Atelectasis is a complete or partial collapse of the entire lung or area (lobe) of the lung. It occurs when the tiny air sacs (alveoli) within the lung become deflated or filled with fluid.",
            "symptoms": "Difficulty breathing, shallow breathing, cough, chest pain, fever.",
            "treatment": "Treatment depends on the cause and may include deep breathing exercises, percussion and vibration techniques, bronchoscopy, or treating underlying conditions."
        },
        "Cardiomegaly": {
            "description": "Cardiomegaly refers to an enlarged heart. It isn't a disease, but rather a sign of another condition that is causing the heart to work harder than normal.",
            "symptoms": "Shortness of breath, especially during activity or when lying flat, swelling in legs, ankles and feet, fatigue, irregular heartbeat.",
            "treatment": "Treatment focuses on the underlying cause and may include medications, surgically implanted devices, or heart surgery."
        },
        "Effusion": {
            "description": "Pleural effusion is the buildup of excess fluid between the layers of the pleura outside the lungs. The pleura are thin membranes that line the lungs and the inside of the chest cavity.",
            "symptoms": "Chest pain, dry cough, fever, difficulty breathing when lying down, shortness of breath.",
            "treatment": "Treatment depends on the cause and severity and may include draining the fluid, medication, or surgery."
        },
        "Infiltration": {
            "description": "Pulmonary infiltrates are substances denser than air, such as pus, blood, or protein, which accumulate in the lungs. They are often seen in pneumonia, tuberculosis, or other infections.",
            "symptoms": "Cough, fever, shortness of breath, chest pain, fatigue.",
            "treatment": "Treatment targets the underlying cause and may include antibiotics, antivirals, or other medications depending on the specific condition."
        },
        "Mass": {
            "description": "A mass in medical imaging can represent different conditions depending on the body part. In chest X-rays, it may represent a lung tumor, abscess, or cancer. In bone X-rays, a mass or abnormal density can indicate a bone fracture, tumor, or other structural abnormality.",
            "symptoms": "Symptoms vary by location. For lungs: cough, shortness of breath, chest pain. For bones: pain, swelling, limited mobility, visible deformity if a fracture is present.",
            "treatment": "Treatment depends on the specific diagnosis and location. It may include surgery, radiation therapy, chemotherapy, or for fractures: immobilization, casting, or surgical repair with pins, plates, or screws."
        },

        "Nodule": {
            "description": "A lung nodule is a small, round growth on the lung that is smaller than 3 cm in diameter. Most lung nodules are benign (not cancerous).",
            "symptoms": "Usually asymptomatic and often discovered incidentally on chest X-rays or CT scans.",
            "treatment": "Often involves monitoring with repeated chest imaging. If suspicious for cancer, biopsy or surgery may be recommended."
        },
        "Pneumonia": {
            "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough, fever, chills, and difficulty breathing.",
            "symptoms": "Cough with phlegm, fever, chills, shortness of breath, chest pain when breathing or coughing, fatigue.",
            "treatment": "Antibiotics for bacterial pneumonia, antivirals for viral pneumonia, cough medicine, pain relievers, and fever reducers."
        },
        "Pneumothorax": {
            "description": "Pneumothorax is a collapsed lung. It occurs when air leaks into the space between the lung and chest wall, causing the lung to collapse partially or completely.",
            "symptoms": "Sudden sharp chest pain, shortness of breath, rapid heart rate, fatigue, bluish skin due to lack of oxygen.",
            "treatment": "Small pneumothoraces may heal on their own. Larger ones may require oxygen therapy, needle aspiration, or chest tube insertion."
        },
        "Consolidation": {
            "description": "Consolidation occurs when the air in the lungs is replaced with a substance, usually fluid. It's often seen in pneumonia and other inflammatory conditions.",
            "symptoms": "Fever, cough, difficult or painful breathing, rapid breathing, shortness of breath.",
            "treatment": "Treatment addresses the underlying cause and may include antibiotics, respiratory therapy, or supportive care."
        },
        "Edema": {
            "description": "Pulmonary edema is fluid accumulation in the tissue and air spaces of the lungs. It can be caused by heart problems, kidney disease, or lung injury.",
            "symptoms": "Shortness of breath, difficulty breathing when lying down, wheezing, cough with pink frothy sputum, anxiety.",
            "treatment": "Treatment targets the underlying cause and may include diuretics, medications to strengthen the heart, oxygen therapy, or mechanical ventilation."
        },
        "Emphysema": {
            "description": "Emphysema is a lung condition that causes shortness of breath. In emphysema, the air sacs (alveoli) in the lungs are damaged, leading to decreased lung function.",
            "symptoms": "Shortness of breath, especially during physical activities, wheezing, chest tightness, chronic cough.",
            "treatment": "Treatment includes smoking cessation, bronchodilators, inhaled steroids, oxygen therapy, and pulmonary rehabilitation."
        },
        "Fibrosis": {
            "description": "Pulmonary fibrosis is a condition in which lung tissue becomes scarred and thickened, making it difficult for the lungs to work properly.",
            "symptoms": "Shortness of breath, a dry, hacking cough, fatigue, unexplained weight loss, aching muscles and joints.",
            "treatment": "Treatment focuses on slowing disease progression and may include medications, oxygen therapy, pulmonary rehabilitation, or lung transplantation."
        },
        "Pleural_Thickening": {
            "description": "Pleural thickening is a lung disease in which the lining of the lungs (pleura) becomes thickened, which can restrict breathing.",
            "symptoms": "Shortness of breath, chest pain, reduced lung function, fatigue.",
            "treatment": "Treatment may include medications to reduce inflammation, pulmonary rehabilitation, and sometimes surgery to remove thickened pleura."
        },
        "Tuberculosis": {
            "description": "Tuberculosis (TB) is a bacterial infection caused by Mycobacterium tuberculosis that primarily affects the lungs. On X-rays, it often appears as cavitations or infiltrates in the upper lobes of the lungs.",
            "symptoms": "Persistent cough (often with blood-tinged sputum), chest pain, weight loss, night sweats, fever, and fatigue.",
            "treatment": "Treatment requires a combination of antibiotics taken for 6-9 months. Multiple drugs are used to prevent antibiotic resistance."
        },
        "Pulmonary_Edema": {
            "description": "Pulmonary edema is a condition caused by excess fluid in the lungs, often due to heart failure. On X-rays, it appears as increased density in the lung fields with a butterfly pattern.",
            "symptoms": "Severe shortness of breath, feeling of drowning or suffocation, wheezing, coughing up frothy sputum, anxiety and restlessness.",
            "treatment": "Treatment includes oxygen therapy, diuretics to remove excess fluid, and medications to treat the underlying cause, often heart failure."
        },
        "COPD": {
            "description": "Chronic Obstructive Pulmonary Disease is a group of progressive lung diseases that includes emphysema and chronic bronchitis. X-rays may show hyperinflated lungs, flattened diaphragm, and increased lung transparency.",
            "symptoms": "Progressive shortness of breath, chronic cough, wheezing, chest tightness, frequent respiratory infections, and fatigue.",
            "treatment": "While not curable, management includes bronchodilators, inhaled steroids, pulmonary rehabilitation, oxygen therapy, and in severe cases, surgery."
        },
        "Interstitial_Lung_Disease": {
            "description": "Interstitial Lung Disease (ILD) refers to a group of disorders affecting the lung interstitium (the tissue and space around the air sacs). X-rays show reticular (net-like) patterns or ground-glass opacities.",
            "symptoms": "Progressive shortness of breath, dry cough, fatigue, weight loss, and clubbing of fingers and toes in advanced cases.",
            "treatment": "Treatment depends on the specific type of ILD and may include corticosteroids, immunosuppressants, antifibrotic medications, and oxygen therapy."
        },
        "No Finding": {
            "description": "No abnormalities were detected in the X-ray image.",
            "symptoms": "No symptoms related to lung or chest conditions were identified in this analysis.",
            "treatment": "No treatment is indicated based on this X-ray as no abnormalities were detected."
        }
    }
    
    # Return info for requested condition, or generic message if not found
    return conditions_info.get(condition, {
        "description": "Information about this condition is not available in our database.",
        "symptoms": "Please consult with a healthcare professional for information about symptoms.",
        "treatment": "Please consult with a healthcare professional for information about treatment options."
    })
