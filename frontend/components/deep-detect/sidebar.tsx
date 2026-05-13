"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import {
  LayoutDashboard,
  Image,
  Brain,
  FileText,
  BookOpen,
  Settings,
  ChevronLeft,
  ChevronRight,
  Shield,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { ThemeToggle } from "./theme-toggle"

interface NavItem {
  icon: React.ElementType
  label: string
  id: string
}

const navItems: NavItem[] = [
  { icon: LayoutDashboard, label: "Ana Panel", id: "dashboard" },
  { icon: Image, label: "Görüntü Analizi", id: "analysis" },
  { icon: Brain, label: "Yapay Zeka Tespiti", id: "detection" },
  { icon: FileText, label: "Raporlar", id: "reports" },
  { icon: BookOpen, label: "Dokümantasyon", id: "docs" },
  { icon: Settings, label: "Ayarlar", id: "settings" },
]

interface SidebarProps {
  activeSection: string
  onSectionChange: (section: string) => void
}

export function Sidebar({ activeSection, onSectionChange }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <motion.aside
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className={cn(
        "fixed left-0 top-0 h-full z-40 glass border-r border-border/50 flex flex-col transition-all duration-300",
        collapsed ? "w-20" : "w-64"
      )}
    >
      {/* Logo */}
      <div className="p-6 border-b border-border/30">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-cyan-400 flex items-center justify-center glow-cyan">
              <Shield className="w-6 h-6 text-primary-foreground" />
            </div>
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-success rounded-full animate-pulse" />
          </div>
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <h1 className="text-xl font-bold text-foreground">
                Deep<span className="text-primary glow-text-cyan">Detect</span>
              </h1>
              <p className="text-xs text-muted-foreground">Yapay Zeka Güvenliği</p>
            </motion.div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item, index) => (
          <motion.button
            key={item.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => onSectionChange(item.id)}
            className={cn(
              "w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-300 group",
              activeSection === item.id
                ? "bg-primary/20 text-primary border border-primary/30 glow-border-cyan"
                : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
            )}
          >
            <item.icon
              className={cn(
                "w-5 h-5 transition-all duration-300",
                activeSection === item.id
                  ? "text-primary"
                  : "group-hover:text-primary"
              )}
            />
            {!collapsed && (
              <span className="font-medium">{item.label}</span>
            )}
            {activeSection === item.id && !collapsed && (
              <motion.div
                layoutId="activeIndicator"
                className="ml-auto w-2 h-2 rounded-full bg-primary"
              />
            )}
          </motion.button>
        ))}
      </nav>

      {/* Theme Toggle */}
      <div className="px-4 pt-2">
        <ThemeToggle collapsed={collapsed} />
      </div>

      {/* Collapse Toggle */}
      <div className="p-4 border-t border-border/30">
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-secondary/50 transition-all"
        >
          {collapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <>
              <ChevronLeft className="w-5 h-5" />
              <span className="text-sm">Daralt</span>
            </>
          )}
        </button>
      </div>

      {/* Status Indicator */}
      {!collapsed && (
        <div className="p-4 border-t border-border/30">
          <div className="glass-card rounded-xl p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
              <span className="text-xs text-success font-medium">Sistem Aktif</span>
            </div>
            <p className="text-xs text-muted-foreground">
              Yapay Zeka Modelleri Aktif: 4/4
            </p>
          </div>
        </div>
      )}
    </motion.aside>
  )
}
