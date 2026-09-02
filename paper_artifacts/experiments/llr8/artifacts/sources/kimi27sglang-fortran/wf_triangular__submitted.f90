subroutine wf_triangular_fp64(a, LEN_2D) bind(c, name='wf_triangular_fp64')
  use iso_c_binding
  implicit none
  integer(c_int64_t), value, intent(in) :: LEN_2D
  real(c_double), intent(inout) :: a(0:LEN_2D-1, 0:LEN_2D-1)
  integer(c_int64_t) :: i, j, istart, iend, jstart, jend
  integer :: ti, tj, nti, ntj, ts_i, ts_j
  integer(1), allocatable :: dep(:,:)

  if (LEN_2D < 2) return

  ts_i = 768
  ts_j = 768
  nti = int((LEN_2D + ts_i - 1) / ts_i)
  ntj = int((LEN_2D + ts_j - 1) / ts_j)

  if (nti < 2 .or. ntj < 2) then
    do i = 1, LEN_2D - 1
      do j = i, LEN_2D - 1
        a(j,i) = a(j,i) + a(j,i-1) + a(j-1,i)
      end do
    end do
    return
  end if

  allocate(dep(0:ntj, 0:nti))

  !$omp parallel shared(dep)
  !$omp single

  do ti = 0, nti
    !$omp task depend(out: dep(0,ti))
      continue
    !$omp end task
  end do

  do tj = 1, ntj
    !$omp task depend(out: dep(tj,0))
      continue
    !$omp end task
  end do

  do ti = 1, nti
    istart = int(ti - 1, c_int64_t) * ts_i
    iend = min(istart + ts_i, LEN_2D) - 1
    istart = max(istart, 1_c_int64_t)
    do tj = 1, ntj
      jstart = int(tj - 1, c_int64_t) * ts_j
      jend = min(jstart + ts_j, LEN_2D) - 1
      !$omp task depend(in: dep(tj-1,ti), dep(tj,ti-1)) depend(out: dep(tj,ti)) &
      !$omp firstprivate(istart,iend,jstart,jend)
        block
          integer(c_int64_t) :: ii, jj
          do ii = istart, iend
            do jj = max(ii, jstart), jend
              a(jj,ii) = a(jj,ii) + a(jj,ii-1) + a(jj-1,ii)
            end do
          end do
        end block
      !$omp end task
    end do
  end do

  !$omp end single
  !$omp end parallel

  deallocate(dep)
end subroutine wf_triangular_fp64
