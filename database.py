from typing import Final, List, Optional

from sqlalchemy import Column, ForeignKey, String, Table, create_engine
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    declared_attr,
    mapped_column,
    relationship,
    sessionmaker,
)

# --------------------------------------
# Create the SQLite database
# --------------------------------------


# Declare the models
class Base(DeclarativeBase):
    pass


class CommonMixin:
    @declared_attr.directive
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    # Define a standard representation of an instance
    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(id={self.id!r})"


class Analysis(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID
    # renewal_year: Mapped[int]
    name: Mapped[str] = mapped_column(String(50))
    # description: Mapped[str] = mapped_column(String(1000))
    # quote: Mapped[int]
    # country: Mapped[str] = mapped_column(String(2))
    # # Country code ISO 3166-1 alpha-2: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
    # treaty_class: Mapped[str] = mapped_column(String(50))
    # # treaty_class = Non-proportional/Proportional
    # currency: Mapped[str] = mapped_column(String(3))
    # exchange_rate_date: Mapped[datetime]
    # state = Mapped[str]
    # status = Mapped[str]
    # source_id: Mapped[int]
    # # source_id = ID of the analysis from which the current analysis was copied, if applicable

    # Define the 1-to-many relationship between Analysis and Layer
    layers: Mapped[List["Layer"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class LayerMixin:
    # # pvse: Mapped[str] = mapped_column(String(21))  # T1E1234-2024-01-01-00
    # # option: Mapped[str] = mapped_column(String(2))  # 01
    # # name: Mapped[str] = mapped_column(String(50))
    # # coverage: Mapped[str] = mapped_column(String(50))  # coverage = Branche fine AGIR
    # # cat_cover: Mapped[bool]
    # premium: Mapped[int]
    occ_limit: Mapped[int]
    occ_deduct: Mapped[int]
    agg_limit: Mapped[int]
    agg_deduct: Mapped[int]
    # brokerage: Mapped[float]
    # taxes: Mapped[float]
    # overriding_commission: Mapped[float]
    # other_loadings: Mapped[float]
    # treaty_rate: Mapped[float]
    # participation: Mapped[float]
    # inception: Mapped[datetime]
    # expiry: Mapped[datetime]
    # currency: Mapped[str] = mapped_column(String(3))
    # category: Mapped[int]
    # # category gets its values in RefCategory.id
    # layer_class: Mapped[int]
    # # layer_class gets its values in RefClass.id
    # exposure_region: Mapped[int]
    # # exposure_region gets its values in RefExposureRegion.id
    # status: Mapped[int]
    # # status gets its values in RefStatus.id
    # display_order: Mapped[int]


# Approx. 10k records per year
class Layer(CommonMixin, LayerMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID

    # Define the 1-to-many relationship between Analysis and Layer
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analysis.id"))
    analysis: Mapped["Analysis"] = relationship(back_populates="layers")

    # Define the 1-to-many relationship between Layer and LayerReinstatement
    reinstatements: Mapped[List["LayerReinstatement"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )

    # Get the premiumfiles associated to the layer through the association layer_premiumfile
    premiumfiles: Mapped[List["PremiumFile"]] = relationship(
        secondary=lambda: layer_premiumfile
    )

    # Get the histolossfiles associated to the layer through the association layer_histolossfile
    histolossfiles: Mapped[List["HistoLossFile"]] = relationship(
        secondary=lambda: layer_histolossfile
    )

    # Get the modelfiles associated to the layer through the association layer_modelfile
    modelfiles: Mapped[List["ModelFile"]] = relationship(
        secondary=lambda: layer_modelfile
    )

    # Define the 1-to-many relationship between Layer and LayerYearLoss
    yearlosses: Mapped[List["LayerYearLoss"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )

    # Define the 1-to-many relationship between Layer and
    burningcosts: Mapped[List["LayerBurningCost"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )


class LayerReinstatementMixin:
    order: Mapped[int]
    number: Mapped[int]  # -1 for unlimited
    rate: Mapped[int]


# Approx. 30k records per year
class LayerReinstatement(CommonMixin, LayerReinstatementMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID

    # Define the 1-to-many relationship between Layer and LayerReinstatement
    layer_id: Mapped[int] = mapped_column(ForeignKey("layer.id"))
    layer: Mapped["Layer"] = relationship(back_populates="reinstatements")


class LayerYearLoss(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    year: Mapped[int]
    day: Mapped[int]
    gross: Mapped[int]
    ceded: Mapped[int]
    net: Mapped[int]
    reinstated: Mapped[int]
    reinst_premium: Mapped[int]
    loss_type: Mapped[str] = mapped_column(String(50))  # Cat/Non-cat
    # peril_id: Mapped[int]
    # peril: Mapped[str] = mapped_column(String(50))
    # model_id: Mapped[int]
    # model: Mapped[str] = mapped_column(String(50))
    # region: Mapped[str] = mapped_column(String(50))
    # line_of_business: Mapped[str] = mapped_column(String(50))

    # Define the 1-to-many relationship between Layer and ResultYearLoss
    layer_id: Mapped[int] = mapped_column(ForeignKey("layer.id"))
    layer: Mapped["Layer"] = relationship(back_populates="yearlosses")


class PremiumFile(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID
    name: Mapped[str] = mapped_column(String(50))
    # data_date: Mapped[datetime]
    # description: Mapped[str] = mapped_column(String(1000))
    # currency: Mapped[str] = mapped_column(String(3))
    # as_if_use_risk_count: Mapped[bool]
    # # as_if_use_risk_count indicates if the risk measure is used for adjusting the as-is premiums to as-if
    # as_if_agir_index: Mapped[Optional[int]]
    # # as_if_agir_index identifies the AGIR price index used for as-ification with an index
    # as_if_inflation_rate: Mapped[Optional[float]]
    # # as_if_inflation_rate is entered if used. E.g. 5% (per year)

    # Define the 1-to-many relationship between PremiumFile and Premium
    premiums: Mapped[List["Premium"]] = relationship(
        back_populates="premiumfile", cascade="all, delete-orphan"
    )


layer_premiumfile: Final[Table] = Table(
    "layer_premiumfile",
    Base.metadata,
    Column("layer_id", ForeignKey("layer.id"), primary_key=True),
    Column("premiumfile_id", ForeignKey("premiumfile.id"), primary_key=True),
)


class Premium(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    year: Mapped[int]
    # risk_count: Mapped[Optional[int]]
    # risk_count = number of policies covered
    as_is_premium: Mapped[int]
    as_if_premium: Mapped[int]

    # Define the 1-to-many relationship between PremiumFile and Premium
    premiumfile_id: Mapped[int] = mapped_column(ForeignKey("premiumfile.id"))
    premiumfile: Mapped["PremiumFile"] = relationship(back_populates="premiums")


class HistoLossFile(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID
    name: Mapped[str] = mapped_column(String(50))
    # description: Mapped[str] = mapped_column(String(1000))
    # currency: Mapped[str] = mapped_column(String(3))
    # data_date: Mapped[datetime]
    # as_is_threshold: Mapped[int]
    start_year: Mapped[int]
    end_year: Mapped[int]
    # as_if_agir_index: Mapped[Optional[int]]
    # # as_if_agir_index = Identifier of the AGIR price index used for the as-ification, if applicable
    # as_if_inflation_rate: Mapped[Optional[float]]
    # as_if_threshold: Mapped[int]

    # Define the 1-to-many relationship between HistoLossFile and HistoLoss
    losses: Mapped[List["HistoLoss"]] = relationship(
        back_populates="lossfile", cascade="all, delete-orphan"
    )


layer_histolossfile: Final[Table] = Table(
    "layer_histolossfile",
    Base.metadata,
    Column("layer_id", ForeignKey("layer.id"), primary_key=True),
    Column("histolossfile_id", ForeignKey("histolossfile.id"), primary_key=True),
)


class HistoLoss(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    year: Mapped[int]
    # loss_type: Mapped[str] = mapped_column(String(50))  # loss_type = Cat/Non-cat
    # name: Mapped[str] = mapped_column(String(50))
    as_is_loss: Mapped[int]
    as_if_loss: Mapped[int]

    # Define the 1-to-many relationship between HistoLossFile and HistoLoss
    lossfile_id: Mapped[int] = mapped_column(ForeignKey("histolossfile.id"))
    lossfile: Mapped["HistoLossFile"] = relationship(back_populates="losses")


class LayerBurningCost(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    basis: Mapped[str] = mapped_column(String(5))
    # basis = As-Is/As-If
    year: Mapped[int]
    year_selected: Mapped[bool]
    # year_selected = True if year is selected for calculating the burning cost, False otherwise
    premium: Mapped[int]
    ceded_before_agg_limits: Mapped[int]
    ceded: Mapped[int]
    ceded_loss_count: Mapped[int]
    reinstated: Mapped[int]

    # Define the 1-to-many relationship between Layer and LayerBurningCost
    layer_id: Mapped[int] = mapped_column(ForeignKey("layer.id"))
    layer: Mapped["Layer"] = relationship(back_populates="burningcosts")


class ModelFile(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID
    # currency: Mapped[str] = mapped_column(String(3))
    # model_class: Mapped[str] = mapped_column(String(50))
    # # model_class (code/digit) = class of the underlying model = Empiricial/Frequency-Severity/Exposure-based/blending/Target premium
    # name: Mapped[str] = mapped_column(String(50))
    # description: Mapped[str] = mapped_column(String(1000))
    # loss_type: Mapped[str] = mapped_column(String(50))  # loss_type = Cat/Non-cat
    years_simulated: Mapped[int]

    # Define the 1-to-many relationship between ModelFile and ModelYearLoss
    yearlosses: Mapped[List["ModelYearLoss"]] = relationship(
        back_populates="modelfile", cascade="all, delete-orphan"
    )


class ModelYearLoss(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    year: Mapped[int]
    day: Mapped[int]
    loss: Mapped[float]
    loss_type: Mapped[str] = mapped_column(String(50))  # Cat/Non-cat
    # peril_id: Mapped[int]
    # peril: Mapped[str] = mapped_column(String(50))
    # model_id: Mapped[int]
    # model: Mapped[str] = mapped_column(String(50))
    # region: Mapped[str] = mapped_column(String(50))
    # line_of_business: Mapped[str] = mapped_column(String(50))

    # Define the 1-to-many relationship between ModelFile and ModelYearLoss
    modelfile_id: Mapped[int] = mapped_column(ForeignKey("modelfile.id"))
    modelfile: Mapped["ModelFile"] = relationship(back_populates="yearlosses")


layer_modelfile: Final[Table] = Table(
    "layer_modelfile",
    Base.metadata,
    Column("layer_id", ForeignKey("layer.id"), primary_key=True),
    Column("modelfile_id", ForeignKey("modelfile.id"), primary_key=True),
)


class ResultInstance(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    # created_on: Mapped[datetime]
    # created_by: Mapped[int]  # User ID
    # modified_on: Mapped[datetime]
    # modified_by: Mapped[int]  # User ID
    name: Mapped[str] = mapped_column(String(50))

    # Define the 1-to-many relationship between ResultInstance and ResultLayer
    layers: Mapped[List["ResultLayer"]] = relationship(back_populates="instance")


class ResultLayer(CommonMixin, LayerMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int]
    # source_id = ID of the layer from which resultlayer was copied

    # Define the 1-to-many relationship between ResultInstance and ResultLayer
    instance_id: Mapped[int] = mapped_column(ForeignKey("resultinstance.id"))
    instance: Mapped["ResultInstance"] = relationship(back_populates="layers")

    # Define the 1-to-many relationship between ResultLayer and ResultLayerReinstatement
    reinstatements: Mapped[List["ResultLayerReinstatement"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )

    # Get the modelfiles associated to the resultlayer through the association resultlayer_modelfile
    modelfiles: Mapped[List["ModelFile"]] = relationship(
        secondary=lambda: resultlayer_modelfile
    )

    # Define the 1-to-many relationship between ResultLayer and ResultLayerStatisticLoss
    percentilelosses: Mapped[List["ResultLayerStatisticLoss"]] = relationship(
        back_populates="layer", cascade="all, delete-orphan"
    )


class ResultLayerReinstatement(CommonMixin, LayerReinstatementMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)

    # Define the 1-to-many relationship between ResultLayer and ResultLayerReinstatement
    layer_id: Mapped[int] = mapped_column(ForeignKey("resultlayer.id"))
    layer: Mapped["ResultLayer"] = relationship(back_populates="reinstatements")


resultlayer_modelfile: Final[Table] = Table(
    "resultlayer_modelfile",
    Base.metadata,
    Column("resultlayer_id", ForeignKey("resultlayer.id"), primary_key=True),
    Column("modelfile_id", ForeignKey("modelfile.id"), primary_key=True),
)


class ResultLayerStatisticLoss(CommonMixin, Base):
    id: Mapped[int] = mapped_column(primary_key=True)
    statistic: Mapped[str] = mapped_column(String(50))
    # statistic = AEP/OEP/Expected Loss Agg/Expected Loss Occ
    percentile: Mapped[Optional[float]]
    loss: Mapped[int]

    # Define the 1-to-many relationship between ResultLayer and ResultLayerStatisticLoss
    layer_id: Mapped[int] = mapped_column(ForeignKey("resultlayer.id"))
    layer: Mapped["ResultLayer"] = relationship(back_populates="percentilelosses")


# Create an engine connected to a SQLite database
engine = create_engine("sqlite://")

# Create the database tables
Base.metadata.create_all(engine)

# Create a session to the database
Session = sessionmaker(engine)
