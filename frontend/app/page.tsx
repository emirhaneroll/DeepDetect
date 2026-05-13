"use client"

import { useState, useCallback } from "react"
import { AnimatePresence } from "framer-motion"
import { ParticleBackground } from "@/components/deep-detect/particle-background"
import { Sidebar } from "@/components/deep-detect/sidebar"
import { HeroSection } from "@/components/deep-detect/hero-section"
import { AnalysisSection } from "@/components/deep-detect/analysis-section"
import { SuspiciousRegion } from "@/components/deep-detect/suspicious-region"
import { RiskAnalysis } from "@/components/deep-detect/risk-analysis"
import { FinalResult } from "@/components/deep-detect/final-result"
import { ReportSection } from "@/components/deep-detect/report-section"
import { LoadingScreen } from "@/components/deep-detect/loading-screen"

// Mock analysis data - in a real app, this would come from your backend
const generateMockAnalysis = () => ({
  analysisData: {
    orb: {
      keypoints: Math.floor(Math.random() * 1500) + 500,
      description: "ORB analizi birden fazla dönüşe duyarsız anahtar nokta tespit etti. Örüntü tutarlılığı, özellik yoğun alanlarda minimum manipülasyon olduğunu gösteriyor.",
    },
    akaze: {
      keypoints: Math.floor(Math.random() * 1200) + 400,
      description: "AKAZE doğrusal olmayan ölçek uzayı özellikleri tespit etti. Kenar koruması özgün sıkıştırma yapaylıklarını gösteriyor.",
    },
    sift: {
      keypoints: Math.floor(Math.random() * 2000) + 800,
      description: "SIFT analizi doğal dağılımlı ölçek değişmez özellikler gösteriyor. Kopyala-yapıştır manipülasyonu belirtisi yok.",
    },
    surf: {
      keypoints: Math.floor(Math.random() * 1800) + 600,
      description: "SURF birden fazla ölçekte güçlü özellikler tespit etti. Blob tespit örüntüleri orijinal çekim ile tutarlı görünüyor.",
    },
  },
  suspiciousRegions: [
    { id: 1, x: 25, y: 30, width: 15, height: 20, confidence: 72 },
    { id: 2, x: 60, y: 45, width: 12, height: 18, confidence: 58 },
  ],
  riskScores: {
    cnn: Math.floor(Math.random() * 40) + 20,
    lstm: Math.floor(Math.random() * 35) + 15,
    final: Math.floor(Math.random() * 45) + 25,
  },
  report: {
    summary: "Analiz edilen görüntü kapsamlı çoklu model adli incelemeden geçirildi. Yapay zeka sistemlerimiz yerelleştirilmiş bölgelerde potansiyel manipülasyonun orta düzey göstergelerini tespit etti. Özellik çıkarma algoritmaları, işlem sonrası değişiklikleri düşündürebilecek piksel düzeyinde örüntü tutarsızlıkları belirledi. Ancak bu bulgular, görüntü kaynağı ve kullanım amacı hakkında ek bağlamla doğrulanmalıdır.",
    technicalDetails: [
      "ORB, AKAZE, SIFT ve SURF algoritmaları kullanılarak özellik çıkarımı tamamlandı",
      "CNN modeli 12 katman boyunca 47 farklı görüntü özelliğini analiz etti",
      "LSTM ağı zamansal tutarlılık için sıralı piksel örüntülerini işledi",
      "Hata Seviyesi Analizi (ELA) birden fazla sıkıştırma seviyesinde gerçekleştirildi",
      "Metadata incelemesi, anomali tespit edilmeyen standart EXIF verilerini ortaya koydu",
    ],
    recommendations: [
      "Temel karşılaştırma için orijinal kaynak dosyayı talep edin",
      "Ters görüntü araması ile görüntü kökenini doğrulayın",
      "Çapraz doğrulama için ek adli araçlar uygulayın",
      "Nihai karar vermeden önce görüntü kullanım bağlamını değerlendirin",
    ],
    timestamp: new Date().toLocaleString("tr-TR"),
    analysisId: `DD-${Date.now().toString(36).toUpperCase()}`,
  },
})

export default function DeepDetectPage() {
  const [activeSection, setActiveSection] = useState("dashboard")
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showResults, setShowResults] = useState(false)
  const [analysisResults, setAnalysisResults] = useState<ReturnType<typeof generateMockAnalysis> | null>(null)

  const handleImageUpload = useCallback((file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string)
      setIsAnalyzing(true)
      setShowResults(false)
    }
    reader.readAsDataURL(file)
  }, [])

  const handleLoadingComplete = useCallback(() => {
    setIsAnalyzing(false)
    setShowResults(true)
    setAnalysisResults(generateMockAnalysis())
    setActiveSection("analysis")
  }, [])

  const getResultType = () => {
    if (!analysisResults) return "original"
    const { final } = analysisResults.riskScores
    if (final < 30) return "original"
    if (final < 60) return "suspicious"
    return "fake"
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Particle Background */}
      <ParticleBackground />

      {/* Loading Screen */}
      <AnimatePresence>
        {isAnalyzing && (
          <LoadingScreen isLoading={isAnalyzing} onComplete={handleLoadingComplete} />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <Sidebar activeSection={activeSection} onSectionChange={setActiveSection} />

      {/* Main Content */}
      <main className="ml-20 md:ml-64 min-h-screen relative z-10 transition-all duration-300">
        <div className="max-w-7xl mx-auto px-4 md:px-8 py-8">
          {/* Hero / Upload Section */}
          {!showResults && (
            <HeroSection onImageUpload={handleImageUpload} isAnalyzing={isAnalyzing} />
          )}

          {/* Results Sections */}
          {showResults && uploadedImage && analysisResults && (
            <>
              {/* Analysis Section */}
              <AnalysisSection
                imageUrl={uploadedImage}
                analysisData={analysisResults.analysisData}
              />

              {/* Suspicious Region Detection */}
              <SuspiciousRegion
                imageUrl={uploadedImage}
                regions={analysisResults.suspiciousRegions}
              />

              {/* AI Risk Analysis */}
              <RiskAnalysis scores={analysisResults.riskScores} />

              {/* Final Result */}
              <FinalResult
                result={getResultType() as "original" | "suspicious" | "fake"}
                confidence={100 - analysisResults.riskScores.final}
                interpretation={
                  getResultType() === "original"
                    ? "Görüntüde önemli bir manipülasyon belirtisi bulunmuyor"
                    : getResultType() === "suspicious"
                    ? "Bazı bölgeler daha fazla inceleme gerektiriyor"
                    : "Yüksek dijital manipülasyon olasılığı tespit edildi"
                }
              />

              {/* Report Section */}
              <ReportSection report={analysisResults.report} />

              {/* Analyze Another Button */}
              <div className="text-center py-12">
                <button
                  onClick={() => {
                    setShowResults(false)
                    setUploadedImage(null)
                    setAnalysisResults(null)
                    setActiveSection("dashboard")
                  }}
                  className="px-8 py-4 rounded-xl bg-gradient-to-r from-primary to-cyan-500 text-primary-foreground font-semibold hover:opacity-90 transition-all shadow-lg hover:shadow-primary/25"
                >
                  Başka Bir Görüntü Analiz Et
                </button>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}
