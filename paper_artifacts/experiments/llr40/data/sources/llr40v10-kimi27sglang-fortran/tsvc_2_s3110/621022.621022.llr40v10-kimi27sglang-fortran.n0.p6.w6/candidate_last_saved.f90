subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s3110_fp64")
  use iso_c_binding
  use omp_lib
  implicit none
  integer(C_INT64_T), value, intent(in) :: LEN_2D
  real(C_DOUBLE), intent(in) :: aa(LEN_2D * LEN_2D)
  real(C_DOUBLE), intent(out) :: bb(2, 2)

  integer(C_INT64_T) :: i, n2, pos, nthreads, tid
  real(C_DOUBLE) :: maxv, chksum
  real(C_DOUBLE), allocatable :: tmaxv(:)
  integer(C_INT64_T), allocatable :: tpos(:)
  real(C_DOUBLE) :: local_max
  integer(C_INT64_T) :: local_pos

  n2 = LEN_2D * LEN_2D
  nthreads = omp_get_max_threads()
  allocate(tmaxv(nthreads), tpos(nthreads))

  !$omp parallel private(i, tid, local_max, local_pos) shared(tmaxv, tpos, maxv)
  tid = omp_get_thread_num() + 1

  ! Pass 1: per-thread local max
  local_max = -huge(1.0_C_DOUBLE)
  !$omp do schedule(static)
  do i = 1, n2
    local_max = max(local_max, aa(i))
  end do
  tmaxv(tid) = local_max

  !$omp barrier
  !$omp single
  maxv = tmaxv(1)
  do i = 2, nthreads
    if (tmaxv(i) > maxv) maxv = tmaxv(i)
  end do
  !$omp end single

  ! Pass 2: per-thread first position of maxv
  local_pos = n2
  !$omp do schedule(static)
  do i = 1, n2
    local_pos = min(local_pos, merge(i - 1_C_INT64_T, n2, aa(i) == maxv))
  end do
  tpos(tid) = local_pos
  !$omp end parallel

  pos = tpos(1)
  do i = 2, nthreads
    if (tpos(i) < pos) pos = tpos(i)
  end do

  chksum = maxv + dble(pos / LEN_2D) + dble(mod(pos, LEN_2D))
  bb(1, 1) = chksum
end subroutine tsvc_2_s3110_fp64
