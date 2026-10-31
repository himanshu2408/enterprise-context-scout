import React from "react";
import { Cpu, Zap, Code2, BookOpen, Layers } from "lucide-react";
import { DomainInfo } from "../types/contracts";
import { MOCK_DOMAINS } from "../api/client";

export interface DomainSelectorProps {
  selectedDomainId: string;
  onDomainChange: (domainId: string) => void;
  domains?: DomainInfo[];
}

export const DomainSelector: React.FC<DomainSelectorProps> = ({
  selectedDomainId,
  onDomainChange,
  domains = MOCK_DOMAINS,
}) => {
  return (
    <div className="card domain-selector-card" data-testid="domain-selector">
      <div className="card-header">
        <div className="card-title">
          <Cpu className="icon icon-emerald" size={18} />
          <span>DomainPack & QLoRA Adapter Slot (ARC-02 & HW-01)</span>
        </div>
        <div className="flex-row-gap">
          <span className="badge badge-emerald">
            <Zap size={12} className="mr-1" /> Hot-Swap: 0.018 ms
          </span>
          <span className="badge badge-zinc">Apple M5 Pro (48GB)</span>
        </div>
      </div>

      <div className="domain-grid" role="radiogroup" aria-label="Domain pack selector">
        {domains.map((domain) => {
          const isSelected = selectedDomainId === domain.domain_id;
          const isCode = domain.domain_id === "trading_code_scout";
          const isQuant = domain.domain_id === "quant_research_scout";

          const IconComponent = isCode ? Code2 : isQuant ? BookOpen : Layers;

          return (
            <div
              key={domain.domain_id}
              data-testid={`domain-card-${domain.domain_id}`}
              className={`domain-card ${isSelected ? "domain-card-active" : ""}`}
              onClick={() => onDomainChange(domain.domain_id)}
              role="radio"
              aria-checked={isSelected}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onDomainChange(domain.domain_id);
                }
              }}
            >
              <div className="domain-card-header">
                <div className="domain-card-icon-title">
                  <IconComponent
                    size={18}
                    className={
                      isSelected
                        ? "icon-emerald"
                        : isCode
                          ? "icon-sky"
                          : isQuant
                            ? "icon-purple"
                            : "icon-amber"
                    }
                  />
                  <span className="domain-card-title">{domain.display_name}</span>
                </div>
                {isSelected && (
                  <span className="badge badge-emerald text-xs">ACTIVE ADAPTER</span>
                )}
              </div>

              <div className="domain-meta-list">
                <div className="meta-row">
                  <span className="meta-label">Base Model:</span>
                  <span className="meta-val font-mono">{domain.base_model_id}</span>
                </div>
                <div className="meta-row">
                  <span className="meta-label">QLoRA Adapter:</span>
                  <span className="meta-val font-mono">{domain.lora_adapter_id}</span>
                </div>
              </div>

              <div className="domain-specs-row">
                <span className="badge badge-sky text-xs">
                  r={domain.qlora_rank}, &alpha;={domain.qlora_alpha}
                </span>
                <span className="badge badge-purple text-xs">
                  mlx:gpu Loss: {domain.training_loss.toFixed(4)}
                </span>
                <span className="badge badge-emerald text-xs">
                  {domain.hot_swap_latency_ms} ms swap
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
