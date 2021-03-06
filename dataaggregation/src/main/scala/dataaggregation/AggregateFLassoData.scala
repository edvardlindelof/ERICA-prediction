package dataaggregation

import akka.actor._
import org.joda.time.DateTime
import sp.EricaEventLogger._
import sp.gPubSub.API_Data.EricaEvent

import scala.collection.mutable

object AggregateFLassoData extends App {
  val system = ActorSystem("DataAggregation")

  system.actorOf(
    Props(new Logger(FLassoStateKeeper, FrequenciesOfNextHour, "FLasso")),
    "EricaEventLogger"
  )
}

object FLassoStateKeeper extends StateKeeper {
  val subStateKeepers = List(
    new RollingFrequencyKeeper(30),
    new RollingFrequencyKeeper(60),
    new RollingFrequencyKeeper(120),
    UntreatedLowPrioPatientsKeeper,
    PatientKindKeeper,
    StaffKeeper
  )

  override def handleEvent(ev: EricaEvent): Unit = {
    subStateKeepers.foreach(stateKeeper => stateKeeper.handleEvent(ev))
  }

  override def state(sampleTime: DateTime): List[(String, Int)] = {
    subStateKeepers.flatMap(stateKeeper => stateKeeper.state(sampleTime))
  }
}

class RollingFrequencyKeeper(rollingMins: Int = 30) extends StateKeeper {

  implicit def stringToDateTime(s: String) = DateTime.parse(s)

  val minsToSaveRollingValues = 200 // needs to be longer than longest rolling time bco timediff between latest event and nextSampleTime
  // TODO add tillsyn/omvårdnad, will be less trivial bco need to check whether its the first tillsyn
  val eventTitles = "Kölapp" :: "Triage" :: "Läkare" :: "Klar" :: Nil
  val eventIncidenceLBs = mutable.Map(eventTitles.map(str => str -> mutable.ListBuffer[DateTime]()):_*)

  override def handleEvent(ev: EricaEvent): Unit = {
    eventTitles.foreach { title => // drop rolling values older than we care about
      val tooOld = (dt: DateTime) => ev.Start.minusMinutes(minsToSaveRollingValues).isAfter(dt)
      eventIncidenceLBs(title) = eventIncidenceLBs(title).dropWhile(tooOld)
    }

    if(eventTitles.contains(ev.Title)) {
      eventIncidenceLBs(ev.Title) += ev.Start
    }
  }

  override def state(sampleTime: DateTime): List[(String, Int)] = {
    val eventIncidences = eventTitles.map { title =>
      val incidence = eventIncidenceLBs(title).filter(_.isAfter(sampleTime.minusMinutes(rollingMins))).length
      title + rollingMins -> incidence
    }
    eventIncidences
  }

}

object FrequenciesOfNextHour extends FutureTeller {
  override def futureState(events: List[EricaEvent]): List[(String, Int)] = {

    implicit def stringToDateTime(s: String) = DateTime.parse(s)

    // TODO add tillsyn/omvårdnad, will be less trivial bco need to check whether its the first tillsyn
    val eventTitles = "Kölapp" :: "Triage" :: "Läkare" :: "Klar" :: Nil
    val eventIncidence = mutable.Map(eventTitles.map(str => "NextHour" + str -> 0):_*)

    val now = events.head.Start
    val nextHourEvents = events.tail.takeWhile(ev => !ev.Start.isAfter(now.plusHours(1)))

    nextHourEvents.foreach { ev =>
      if(eventTitles.contains(ev.Title)) eventIncidence("NextHour" + ev.Title) += 1
    }

    return eventIncidence.toList
  }
}
