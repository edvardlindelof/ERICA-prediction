package dataaggregation

import akka.actor._
import org.joda.time.DateTime

import sp.EricaEventLogger._
import sp.gPubSub.API_Data.EricaEvent

// the generated data is enough to do something quite similar to the QLasso-article
// the biggest thing that is missing is information of the priorities of the patients at each stage
object AggregateQLassoData extends App {
  val system = ActorSystem("DataAggregation")

  system.actorOf(
    Props(new Logger(QLassoStateKeeper, TTLOfNextLowPrioPatient, "QLasso")),
    "EricaEventLogger"
  )
}

object QLassoStateKeeper extends StateKeeper {
  val subStateKeepers = List(
    UntreatedLowPrioPatientsKeeper,
    new RollingWaitTimeKeeper(30),
    new RollingWaitTimeKeeper(60),
    new RollingWaitTimeKeeper(120),
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

import scala.collection.mutable

object UntreatedLowPrioPatientsKeeper extends StateKeeper {
  val utreatedLowPrioIds = mutable.Set[Int]()

  override def handleEvent(ev: EricaEvent): Unit = {
    if (ev.Type == "PRIO3" || ev.Type == "PRIO4" || ev.Type == "PRIO5") utreatedLowPrioIds.add(ev.CareContactId)
    if (ev.Type == "LÄKARE" || ev.Type == "KLAR" || ev.Category == "RemovedPatient") utreatedLowPrioIds.remove(ev.CareContactId)
  }

  override def state(sampleTime: DateTime): List[(String, Int)] = List(("UntreatedLowPrio", utreatedLowPrioIds.size))
}
