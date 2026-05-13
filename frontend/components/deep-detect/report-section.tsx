"use client"

import { motion } from "framer-motion"
import { FileText, Download, Share2, Copy, Check, Printer } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useState } from "react"

interface ReportData {
  summary: string
  technicalDetails: string[]
  recommendations: string[]
  timestamp: string
  analysisId: string
}

interface ReportSectionProps {
  report: ReportData
}

export function ReportSection({ report }: ReportSectionProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(report.summary)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="py-12"
    >
      <div className="mb-8 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-2 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center">
              <FileText className="w-5 h-5 text-primary" />
            </div>
            Analiz Raporu
          </h2>
          <p className="text-muted-foreground">
            Kapsamlı teknik döküm ve öneriler
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={handleCopy}
            className="glass-card border-border/30 hover:bg-primary/10 hover:text-primary hover:border-primary/30"
          >
            {copied ? (
              <Check className="w-4 h-4 mr-2" />
            ) : (
              <Copy className="w-4 h-4 mr-2" />
            )}
            {copied ? "Kopyalandı!" : "Kopyala"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="glass-card border-border/30 hover:bg-primary/10 hover:text-primary hover:border-primary/30"
          >
            <Printer className="w-4 h-4 mr-2" />
            Yazdır
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="glass-card border-border/30 hover:bg-primary/10 hover:text-primary hover:border-primary/30"
          >
            <Share2 className="w-4 h-4 mr-2" />
            Paylaş
          </Button>
          <Button className="bg-gradient-to-r from-primary to-cyan-500 text-primary-foreground hover:opacity-90">
            <Download className="w-4 h-4 mr-2" />
            PDF Olarak İndir
          </Button>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Main Summary */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2 glass-card rounded-2xl p-6 border border-border/30"
        >
          <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            Yönetici Özeti
          </h3>
          <p className="text-muted-foreground leading-relaxed mb-6">
            {report.summary}
          </p>

          <div className="border-t border-border/20 pt-6">
            <h4 className="text-sm font-semibold text-foreground mb-3 uppercase tracking-wider">
              Teknik Detaylar
            </h4>
            <ul className="space-y-3">
              {report.technicalDetails.map((detail, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className="flex items-start gap-3 text-sm text-muted-foreground"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                  {detail}
                </motion.li>
              ))}
            </ul>
          </div>
        </motion.div>

        {/* Side Panel */}
        <div className="space-y-6">
          {/* Recommendations */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-card rounded-2xl p-6 border border-border/30"
          >
            <h3 className="text-lg font-semibold text-foreground mb-4">
              Öneriler
            </h3>
            <ul className="space-y-3">
              {report.recommendations.map((rec, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 + index * 0.1 }}
                  className="flex items-start gap-3"
                >
                  <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs text-primary font-semibold">
                      {index + 1}
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground">{rec}</p>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* Metadata */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="glass-card rounded-2xl p-6 border border-border/30"
          >
            <h3 className="text-lg font-semibold text-foreground mb-4">
              Rapor Bilgileri
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Analiz ID</span>
                <span className="text-sm font-mono text-foreground bg-secondary/50 px-2 py-1 rounded">
                  {report.analysisId}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Oluşturulma</span>
                <span className="text-sm text-foreground">{report.timestamp}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Yapay Zeka Modelleri</span>
                <span className="text-sm text-foreground">4 Aktif</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">İşlem Süresi</span>
                <span className="text-sm text-foreground">2.3 saniye</span>
              </div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Footer Note */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="mt-6 p-4 rounded-xl bg-primary/5 border border-primary/20"
      >
        <p className="text-xs text-muted-foreground text-center">
          Bu rapor DeepDetect yapay zeka analiz sistemi tarafından oluşturulmuştur. Sonuçlar kritik uygulamalar için uzman incelemesiyle doğrulanmalıdır.
          Bu analiz hakkındaki sorularınız için destek ekibimizle iletişime geçin.
        </p>
      </motion.div>
    </motion.section>
  )
}
